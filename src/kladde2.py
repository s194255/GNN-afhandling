import wandb

# Logg inn på Weights and Biases-kontoen din
wandb.login()

# Hent informasjon om alle kjøringer (runs) i et prosjekt
# runs = wandb.Api().runs("your_username/your_project")
runs = wandb.Api().runs("afhandling")

# Loop gjennom hver kjøring og få tilgang til relevant informasjon
for run in runs:
    print("Run ID:", run.id)
    print("Run Name:", run.name)
    print("Run State:", run.state)
    print("Run Created Time:", run.created_at)
    # print("Run Duration:", run.duration)
    print("Run Tags:", run.tags)
    print("Run Config:", run.config)
    print("Run Metrics:", run.summary_metrics)
    print("---------------")