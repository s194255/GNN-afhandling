import wandb
import src.models as m
import os
import torch

# Logg inn på Weights and Biases-kontoen din
# wandb.login()

# Hent informasjon om alle kjøringer (runs) i et prosjekt
# runs = wandb.Api().runs("your_username/your_project")
# runs = wandb.Api().runs("afhandling")


# Loop gjennom hver kjøring og få tilgang til relevant informasjon
# for run in runs:
    # print("Run ID:", run.id)
    # print("Run Name:", run.name)
    # print("Run State:", run.state)
    # print("Run Created Time:", run.created_at)
    # # print("Run Duration:", run.duration)
    # print("Run Tags:", run.tags)
    # print("Run Config:", run.config)
    # print("Run Metrics:", run.summary_metrics)
    # print("---------------")


# lol = wandb.restore(run_path='s194255/afhandling/model-cbw60c7v:v0', name='model.ckpt')
# a = 2

# run = wandb.init()
# artifact = run.use_artifact('s194255/afhandling/model-cbw60c7v:v0', type='model')
# artifact_dir = artifact.download()
# model = m.Selvvejledt.load_from_checkpoint(os.path.join(artifact_dir, "model.ckpt"))
# a = 2
# run.finish()


# api = wandb.Api()
# artefakt_reference = "s194255/afhandling/model-cbw60c7v:v0"
# artefakt = api.artifact(artefakt_reference)
# artefakt_dir = artefakt.download()
# print("Artefakt downloadet til:", artefakt_dir)
# model = m.Selvvejledt.load_from_checkpoint(os.path.join(artefakt_dir, "model.ckpt"))
# a = 2
n_klasser = 10
n_forkert = n_klasser-1
rigtig = 1.0
forkert = (1-rigtig)/(n_forkert)
pred = torch.tensor([rigtig]+[forkert]*n_forkert).unsqueeze(0)
print(pred)
target = torch.tensor([0], dtype=torch.int64)
ce = torch.nn.functional.cross_entropy(pred, target)
print(ce.item())
print(torch.exp(-ce))