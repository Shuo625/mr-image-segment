from datetime import datetime


def get_cur_time(fmt='%Y-%m-%d %H:%M') -> datetime:
    return datetime.now().strftime(fmt)


class TrainTimeHelper(object):
    def __init__(self, total_epochs: int, total_iters: int):
        self.time = datetime.now()

        self.total_epochs = total_epochs
        self.total_iters = total_iters

    def get_cur_time(self, fmt='%Y-%m-%d %H:%M') -> datetime:
        return datetime.now().strftime(fmt)

    def _get_rest_iters(self, epoch: int, iter: int) -> int:
        return (self.total_epochs - epoch ) * self.total_iters + self.total_iters - iter

    def _get_past_iters(self, epoch: int, iter: int) -> int:
        return (epoch - 1) * self.total_iters + iter

    def rest_time(self, epoch: int, iter: int) -> str:
        last_time = self.time
        self.time = datetime.now()

        rest_time = (self.time - last_time) * self._get_rest_iters(epoch, iter) / self._get_past_iters(epoch, iter) 
        rest_days = rest_time.days
        rest_seconds = rest_time.seconds
        rest_hours = rest_seconds // 3600
        rest_minutes = (rest_seconds // 60) % 60
        rest_seconds = rest_seconds - 3600 * rest_hours - 60 * rest_minutes

        return f'{rest_days}d:{rest_hours}h:{rest_minutes}m:{rest_seconds}s'
