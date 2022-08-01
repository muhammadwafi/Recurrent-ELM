import math


class TfIdf:
    def __init__(self, datas, fitur):
        self.datas = datas
        self.fitur = fitur
        self.tf = self.getTf()
        # print("==> tf ==>",self.tf)
        self.wtd = self.getWtd()
        # print("==> wtf ==>", self.wtd)
        self.idf = self.getIdf()
        # print("==> idf ==>", self.idf)
        self.tfidf = self.tf_idf()
        # print("==> tfidf ==>", self.tfidf)

    def getTf(self):
        return [
            [
                line.count(fitur) for fitur in self.fitur
            ] for line in self.datas
        ]

    def getWtd(self):
        return [
            [
                1 + math.log10(self.tf[i][j]) if self.tf[i][j] > 0 else 0
                for j in range(len(self.tf[i]))
            ]
            for i in range(len(self.tf))
        ]

    def getIdf(self):
        temp_idf = [0 for x in range(len(self.tf[0]))]
        for i in range(len(self.tf)):
            for j in range(len(self.tf[i])):
                if self.tf[i][j] > 0:
                    temp_idf[j] += 1
        return [
            math.log10(len(self.tf) / x) if x != 0 else 0 for x in temp_idf]

    def tf_idf(self):
        return [
            [(self.wtd[i][x] * self.idf[x]) for x in range(len(self.wtd[0]))]
            for i in range(len(self.wtd))
        ]

    def normalisasi(self, tfIdf):
        pembagi = []
        hasilNorm = [[]]
        for i in range(len(tfIdf)):
            tempPembagi = 0
            for j in range(len(tfIdf[i])):
                tempPembagi += pow(tfIdf[i][j], 2)
            pembagi.append(math.sqrt(tempPembagi))

        hasilNorm = [
            [
                0 for i in range(len(tfIdf[l]))
            ] for l in range(len(tfIdf))
        ]

        for k in range(len(tfIdf)):
            for l in range(len(tfIdf[k])):
                if pembagi[k] == 0:
                    hasilNorm[k][l] = 0
                else:
                    hasilNorm[k][l] = tfIdf[k][l] / pembagi[k]
        return hasilNorm

    def getTfIdf(self):
        return self.normalisasi(self.tfidf)
