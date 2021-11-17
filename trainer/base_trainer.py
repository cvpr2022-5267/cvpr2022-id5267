import torch
from abc import abstractmethod
from torch.utils.tensorboard.writer import SummaryWriter
# from tensorboardX import SummaryWriter   # use this if pytorch <= 1.2


def func(m):
    m = torch.nn.DataParallel(m, device_ids=[0, 1])
    return m


class BaseTrainer:
    """
    Base Trainer
    """
    def __init__(self, model, loss, optimizer, config):
        self.best_loss = {'validation': 10000, 'test': 10000}
        self.config = config
        self.model = model
        self.loss = loss
        self.optimizer = optimizer

        self.start_iter = 1
        self.iterations = config.iterations
        assert config.save_path is not None
        self.save_path = config.save_path

        # setup visualization writer instance
        self.writer = SummaryWriter(self.save_path, max_queue=10)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        example return:
            result = {
                      'loss':  0.0,
                      'metric': 1.0,
                      }
        """
        raise NotImplementedError

    @abstractmethod
    def train(self):
        """ Main training loop contained
        E.g.
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)
            printing ...(result)

            if epoch % self.save_period == 0:
                    self._save_checkpoint(epoch)
        """
        raise NotImplementedError

    def _prepare_device(self, use_gpu=[]):
        raise NotImplementedError

    def save_intermediate_model(self, iter):
        raise NotImplementedError

    def _save_checkpoint(self, epoch):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        raise NotImplementedError

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        raise NotImplementedError
