class MetricTracker:
    def __init__(self):
        self.max_nmi = 0
        self.max_ari = 0
        self.max_acc = 0
        self.max_pur = 0

    def update(self, nmi, ari, acc, pur):
        if acc > self.max_acc:
            self.max_nmi = nmi
            self.max_ari = ari
            self.max_acc = acc
            self.max_pur = pur
            return True
        return False

    def get_result(self):
        return self.max_nmi, self.max_ari, self.max_acc, self.max_pur