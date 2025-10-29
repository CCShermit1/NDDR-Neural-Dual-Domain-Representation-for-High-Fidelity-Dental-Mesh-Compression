import os


class Exporter:
    path_self = 'results/'
    results = os.path.join(path_self)
    quadrangulated = os.path.join(path_self, 'quadrangulated')
    torched = os.path.join(path_self, 'torched')
    binaries = os.path.join(path_self, 'binaries')
    loss = os.path.join(path_self, 'loss')
    stl = os.path.join(path_self, 'stl')
    meta = os.path.join(path_self, 'meta')

    @staticmethod
    def dirfill():
        os.makedirs(Exporter.results, exist_ok=True)
        os.makedirs(Exporter.quadrangulated, exist_ok=True)
        os.makedirs(Exporter.torched, exist_ok=True)
        os.makedirs(Exporter.binaries, exist_ok=True)
        os.makedirs(Exporter.loss, exist_ok=True)
        os.makedirs(Exporter.stl, exist_ok=True)
        os.makedirs(Exporter.meta, exist_ok=True)

    def __init__(self, mesh: str, lod: int, features: int):
        Exporter.dirfill()

        self.prefix = os.path.basename(mesh)
        self.prefix = self.prefix.split('.')[0]
        self.basename = self.prefix + f'-lod{lod}-f{features}'

    def partitioned(self):
        return os.path.join(Exporter.quadrangulated, self.basename + '.obj')

    def pytorch(self):
        return os.path.join(Exporter.torched, self.basename + '.pt')

    def binary(self):
        return os.path.join(Exporter.binaries, self.basename + '.bin')

    def plot(self, suffix=""):
        filename = self.basename
        if suffix: 
            filename += suffix
        return os.path.join(Exporter.loss, filename + '.pdf')

    def mesh(self):
        return os.path.join(Exporter.stl, self.basename + '.stl')

    def metadata(self):
        return os.path.join(Exporter.meta, self.basename + '.json')
