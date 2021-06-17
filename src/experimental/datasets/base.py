import torch


class EditDataset(torch.utils.data.Dataset):
    """
    An abstract dataset class which supports generating edits.
    All subclasses must implement edit_generator.
    """

    def edit_generator(self, batch_size):
        """
        Creates and returns an edit generator.
        """
        raise NotImplementedError

