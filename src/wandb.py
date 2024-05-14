from config import CFG
import wandb
import warnings
warnings.filterwarnings("ignore")


def wandb_init():
    if CFG.use_wandb:
        wandb.login(key="c465dd55c08ec111e077cf0454ba111b3a764a78")
        run = wandb.init(
            project=f"{CFG.backbone_model.split('/')[-1]}-{CFG.train_type}",
            job_type="training",
            anonymous="allow"
        )
    else:
        pass
