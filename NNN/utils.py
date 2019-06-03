import warnings
import collections

class Data(object):
    def __init__(self, train, target, batch_size):
        if len(train) != len(target):
            raise ValueError("Num records in train and target need to be the same.")
        self.train = train
        self.target = target
        self.batch_size = batch_size
        self._mini_batch = None

    @classmethod
    def from_dataframe(cls, train, target, batch_size):
        raise NotImplementedError
        #return Data(train, target, batch_size)

    def batch_generator(self):
        if self.batch_size > len(self.train):
            warnings.warn("Batchsize greater than records in data, returning all records per batch.")
        while True:
            for i in range(0, len(self.train), self.batch_size):
                yield Data(self.train[i:i+self.batch_size], self.target[i:i+self.batch_size], self.batch_size)
    @property
    def mini_batch(self):
        if not self._mini_batch:
            self._mini_batch = self.batch_generator()
            return self._mini_batch
        else:
            return self._mini_batch
    
if __name__ == "__main__":
    a = Data(list(range(0,100)), list(range(0,100)), 2)
    print(a)
    