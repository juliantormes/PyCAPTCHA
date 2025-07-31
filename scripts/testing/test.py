from model.model import captcha_model, model_conv, model_resnet
from data.datamodule import captcha_dm
from utils.arg_parsers import test_arg_parser
import pytorch_lightning as pl
import torch

def test(args):
    dm = captcha_dm()
    model = captcha_model.load_from_checkpoint(args.ckpt, model=model_resnet())
    tb_logger = pl.loggers.TensorBoardLogger(
        args.log_dir, name=args.test_name, version=2, default_hp_metric=False)
    
    # Auto-detect device
    if torch.cuda.is_available():
        devices = "auto"
        accelerator = "gpu"
    else:
        devices = "auto"
        accelerator = "cpu"
    
    trainer = pl.Trainer(deterministic=True,
                         devices=devices,
                         accelerator=accelerator,
                         precision='32-true',
                         logger=tb_logger,
                         fast_dev_run=False,
                         max_epochs=5,
                         log_every_n_steps=50,
                         )
    trainer.test(model, dm)

if __name__ == "__main__":
    args = test_arg_parser()
    test(args)
