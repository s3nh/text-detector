import redis 

class WatchDogBase():
    def __init__(self, host = 'localhost', port = 6379, db = 13):
        self.host = host
        self.port = port
        self.db = db 

    def create_base(self):
        r = redis.StrictRedis(db = self.db)
        return r

    def update_base(self):
        pass

    def update_table(self, records):
        pass


    def flush_all(self):
        r.flushall()

    def flush_db(self):
        r.flushdb()


def main():

    wb = WatchDogBase()
    base = wb.create_base()



if __name__ == "__main__":
    main()
