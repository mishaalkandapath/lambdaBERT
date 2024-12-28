import threading

class ThreadLockDict:
    def __init__(self):
        self.lock = threading.Lock()
        self.dict = {}
    
    def __getitem__(self, key):
        with self.lock:
            return self.dict[key]
    
    def __setitem__(self, key, value):
        with self.lock:
            self.dict[key] = value
    
    def __delitem__(self, key):
        with self.lock:
            del self.dict[key]
    
    def __contains__(self, key):
        with self.lock:
            return key in self.dict
    
    def __len__(self):
        with self.lock:
            return len(self.dict)
    
    def __iter__(self):
        with self.lock:
            return iter(self.dict)