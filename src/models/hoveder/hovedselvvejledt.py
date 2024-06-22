import lightning as L
import torch
from torch import Tensor
from torch.autograd import grad
from src.models.hoveder.fælles import GatedEquivariantMotor


class HovedSelvvejledtKlogt(L.LightningModule):
    def __init__(self,
                 n_noise_trin: int,
                 hidden_channels: int,
                 num_layers: int,
                 beregn_lokalt: bool,
                 beregn_globalt: bool,
                 reduce_op: str = "sum",
                 ):
        super().__init__()
        self.n_noise_trin = n_noise_trin
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.reduce_op = reduce_op
        self.beregn_lokalt = beregn_lokalt
        self.beregn_globalt = beregn_globalt


        self.lokal_motor = GatedEquivariantMotor(
            hidden_channels=self.hidden_channels,
            out_channels=1,
            means=torch.zeros(size=(1,)),
            stds=torch.ones(size=(1,)),
            num_layers=self.num_layers,
            reduce_op=self.reduce_op
        )
        self.global_motor = self.create_global_motor()
        self.derivative = True
        self.criterion_globalt = torch.nn.CrossEntropyLoss()
        self.criterion_lokalt = torch.nn.MSELoss(reduction='none')
        self.reset_parameters()

    def create_global_motor(self):
        return GatedEquivariantMotor(
            hidden_channels=self.hidden_channels*2,
            out_channels=self.n_noise_trin,
            means=torch.zeros(size=(self.n_noise_trin,)),
            stds=torch.ones(size=(self.n_noise_trin,)),
            num_layers=self.num_layers,
            reduce_op=self.reduce_op
        )

    def reset_parameters(self):
        self.lokal_motor.reset_parameters()
        self.global_motor.reset_parameters()

    def forward(self, graph_noisy: dict, graph_normal: dict, noise_idx, noise_scale, target):
        loss_lokalt = torch.tensor(0.0, device=self.device)
        loss_globalt = torch.tensor(0.0, device=self.device)
        if self.beregn_lokalt:
            pred_lokalt = self.lokal_motor(**{nøgle: værdi for nøgle, værdi in graph_noisy.items() if nøgle in ['batch', 'x', 'v']})
            z, pos, batch = graph_noisy['z'], graph_noisy['pos'], graph_noisy['batch']
            loss_lokalt = self.get_loss_lokalt(z, pos, batch, pred_lokalt, noise_scale, target)
        if self.beregn_globalt:
            graph_combined = {**{'batch': graph_noisy['batch']}, **{key: torch.cat([graph_noisy[key], graph_normal[key]], dim=-1)
                                                   for key in ['x', 'v']}}
            pred_globalt = self.global_motor(**graph_combined)
            loss_globalt = self.get_loss_globalt(pred_globalt, noise_idx, noise_scale)
        tabsopslag = {}
        tabsopslag['lokalt'] = loss_lokalt
        tabsopslag['globalt'] = loss_globalt
        return tabsopslag

    def get_loss_lokalt(self, z, pos, batch, pred_lokalt, noise_scale, target):
        grad_outputs = [torch.ones_like(pred_lokalt)]
        dy = grad(
            [pred_lokalt],
            [pos],
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
        )[0]
        if dy is None:
            raise RuntimeError(
                "Autograd returned None for the force prediction.")
        noise_scale = torch.gather(noise_scale, 0, batch)
        pred = 1 / noise_scale.view(-1, 1) * dy
        # print(pred.shape, target.shape)
        loss = noise_scale ** 2 * self.criterion_lokalt(pred, target).sum(dim=1)
        loss = loss.mean()
        return loss

    def get_loss_globalt(self, pred_globalt: torch.Tensor, noise_idx: torch.Tensor, noise_scale: torch.Tensor):
        return self.criterion_globalt(pred_globalt, noise_idx)

    @property
    def tabsnøgler(self):
        return ['lokalt', 'globalt']

    # @property
    # def out_channels(self):
    #     return self.n_noise_trin+1

class HovedSelvvejledtKlogtReg(HovedSelvvejledtKlogt):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion_globalt = torch.nn.MSELoss()

    def create_global_motor(self):
        return GatedEquivariantMotor(
            hidden_channels=self.hidden_channels * 2,
            out_channels=1,
            means=torch.zeros(size=(1,)),
            stds=torch.ones(size=(1,)),
            num_layers=self.num_layers,
            reduce_op=self.reduce_op
        )

    def get_loss_globalt(self, pred_globalt: torch.Tensor, noise_idx: torch.Tensor, noise_scale: torch.Tensor):
        return self.criterion_globalt(pred_globalt.squeeze(1), torch.log10(noise_scale))



class HovedSelvvejledtDumt(HovedSelvvejledtKlogt):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        raise NotImplementedError


