import os
import json
import tqdm
import torch
import logging
import nddrutil
import argparse
import pymeshlab
import trimesh
import multiprocessing

from util import *
from source.nddr import NDDR
from render import Renderer

import matplotlib.pyplot as plt
from util.normal_loss import compute_normal_loss, compute_loss_in_batches

from knn import knn_indices

import pandas as pd


class Trainer:
    @staticmethod
    def quadrangulate_surface(mesh: str, count: int, destination: str) -> None:
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(mesh)
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=count, qualitythr=1.0)
        ms.meshing_repair_non_manifold_edges()
        ms.meshing_tri_to_quad_by_smart_triangle_pairing()
        ms.save_current_mesh(destination)
        logging.info(f'Quadrangulated mesh into {destination}')

    def __init__(self, mesh: str, lod: int, features: int, batch: int):
        # Properties
        self.path = os.path.abspath(mesh)
        self.cameras = 200
        self.batch = batch
        self.losses = {}

        logging.info('Launching training process with configuration:')
        logging.info(f'    Reference mesh: {self.path}')
        logging.info(f'    Camera count:   {self.cameras}')
        logging.info(f'    Batch size:     {self.batch}')

        self.exporter = Exporter(mesh, lod, features)

        self.target, normalizer = load_mesh(mesh)
        self.gt_vertices = self.target.vertices.cuda()
        self.gt_faces    = self.target.faces.int().cuda()
        self.gt_normals  = vertex_normals(self.gt_vertices, self.gt_faces)
        logging.info(f'Loaded reference mesh {mesh}')

        qargs = (mesh, 2 * lod, self.exporter.partitioned())
        proc = multiprocessing.Process(target=Trainer.quadrangulate_surface, args=qargs)
        proc.start()

        # Wait for minute before termimating
        proc.join(600)
        if proc.is_alive():
            logging.error('Quadrangulation running overtime')
            proc.terminate()
            exit()

        self.renderer = Renderer()
        logging.info('Constructed renderer for optimization')

        self.views = None
        self.reference_views = None

        self.nddr = NDDR.from_base(self.exporter.partitioned(), normalizer, features)
        

    def precompute_reference_views(self):
        vertices = self.target.vertices
        vertices = vertices[self.target.faces].reshape(-1, 3)
        faces = torch.arange(vertices.shape[0])
        faces = faces.int().cuda().reshape(-1, 3)
        normals = vertex_normals(vertices, faces)

        cache = []
        for view in tqdm.tqdm(self.views, ncols=50, leave=False):
            reference_view = self.renderer.render(vertices, normals, faces, view.unsqueeze(0))
            cache.append(reference_view)

        return list(torch.cat(cache).split(self.batch))

    def optimize_resolution(self, optimizer: torch.optim.Optimizer, rate: int) -> dict[str, list[float]]:
        import numpy as np
        losses = {
            'render': [],
            'laplacian': [],
            'boundary':[],
            'normal': []
        }

        base = self.nddr.base(rate).detach()
        cmap = make_cmap(self.nddr.complexes, self.nddr.points.detach(), base, rate)
        remap = nddrutil.generate_remapper(self.nddr.complexes.cpu(), cmap, base.shape[0], rate)
        quads = torch.from_numpy(quadify(self.nddr.complexes.shape[0], rate)).int()
        graph = nddrutil.Graph(remap.remap(quads), base.shape[0])

        batched_views = list(self.views.split(self.batch))
        length = average_edge_length(base, quads)
        
        # --- Build shared-edge index pairs for boundary continuity (once per rate) ---
        def build_shared_edge_pairs(complexes: torch.Tensor, rate: int, device: str = 'cuda'):
            import numpy as np
            C = complexes.detach().cpu().numpy()
            edge_map = {}
            for q in range(C.shape[0]):
                c0, c1, c2, c3 = int(C[q,0]), int(C[q,1]), int(C[q,2]), int(C[q,3])
                local_edges = [
                    (c0, c1, 0),
                    (c1, c2, 1),
                    (c2, c3, 2),
                    (c3, c0, 3)
                ]
                for a, b, eid in local_edges:
                    key = (min(a,b), max(a,b))
                    edge_map.setdefault(key, []).append((q, eid, (a,b)))

            idx_a, idx_b = [], []
            ar = torch.arange(rate, device=device, dtype=torch.int64)
            seq = {
                0: (lambda base: base + ar * rate + 0),                
                1: (lambda base: base + (rate - 1) * rate + ar),       
                2: (lambda base: base + ar.flip(0) * rate + (rate - 1)),
                3: (lambda base: base + 0 * rate + ar.flip(0))          
            }

            for key, items in edge_map.items():
                if len(items) != 2:
                    continue  
                (qa, ea, oa), (qb, eb, ob) = items
                base_a = qa * rate * rate
                base_b = qb * rate * rate
                ida = seq[ea](torch.tensor(base_a, device=device, dtype=torch.int64))
                idb = seq[eb](torch.tensor(base_b, device=device, dtype=torch.int64))
                if oa[0] == ob[1] and oa[1] == ob[0]:
                    idb = idb.flip(0)
                idx_a.append(ida)
                idx_b.append(idb)

            if len(idx_a) == 0:
                return None, None
            return torch.cat(idx_a).long(), torch.cat(idx_b).long()

        boundary_idx_a, boundary_idx_b = build_shared_edge_pairs(self.nddr.complexes, rate)
        

        for _ in tqdm.trange(100, ncols=50, leave=False):
            batch_losses = {
                'render': [],
                'laplacian': [],
                'boundary':[],
                'normal': []
            }

            batch_losses = {
                'render': [],
                'laplacian': [],
                'boundary':[],
                'normal':[]
                
            }

            uvs = self.nddr.sampler(rate)
            uniform_uvs = self.nddr.sample_uniform(rate)

            for batch_views, ref_views in zip(batched_views, self.reference_views):
                vertices = self.nddr.eval(*uvs)
                uniform_vertices = self.nddr.eval(*uniform_uvs)

                faces = nddrutil.triangulate_shorted(vertices, self.nddr.complexes.shape[0], rate)
                faces = remap.remap_device(faces)

                vertices, normals, faces = separate(vertices, faces)
                smoothed_vertices = graph.smooth(uniform_vertices, 1.0)
                smoothed_vertices = remap.scatter_device(smoothed_vertices)
                laplacian_loss = (uniform_vertices - smoothed_vertices).abs().mean()

                K_sample = 5000
                perm = torch.randperm(vertices.size(0), device=vertices.device)[:K_sample]
                verts_s   = vertices[perm]
                normals_s = normals[perm]
                idx = knn_indices(verts_s, self.gt_vertices, k=1, chunk=2000)
                matched_normals = self.gt_normals[idx.squeeze(-1)]
                nc_loss_uniform = 1.0 - torch.sum(normals_s * matched_normals, dim=-1).clamp(-1.0, 1.0)
                nc_loss_uniform = nc_loss_uniform.mean()

                patches = self.nddr.complexes.shape[0]
                grid = uniform_vertices.view(patches, rate, rate, 3)

                def shift(t, dim, offset):
                    pads = [0, 0, 0, 0, 0, 0, 0, 0]
                    idx = torch.arange(t.size(dim), device=t.device)
                    idx = (idx.to(torch.int64) + offset).clamp(0, t.size(dim)-1)
                    indexer = [slice(None)] * t.dim()
                    indexer[dim] = idx
                    return t[tuple(indexer)]

                du = 0.5 * (shift(grid, 1, +1) - shift(grid, 1, -1))
                dv = 0.5 * (shift(grid, 2, +1) - shift(grid, 2, -1))
                n_grid = torch.linalg.cross(du, dv, dim=-1)
                n_norm = torch.linalg.norm(n_grid, dim=-1, keepdim=True).clamp_min(1e-8)
                n_grid = n_grid / n_norm

                center = grid
                lap_u = shift(grid, 1, +1) + shift(grid, 1, -1) - 2 * center
                lap_v = shift(grid, 2, +1) + shift(grid, 2, -1) - 2 * center
                curvature = torch.linalg.norm(lap_u, dim=-1) + torch.linalg.norm(lap_v, dim=-1)  # [P,R,R]

                weights = curvature.reshape(-1)
                weights = (weights - weights.min()).clamp_min(0)
                if float(weights.sum().item()) == 0.0:
                    weights = torch.ones_like(weights)

                K_adapt = K_sample  
                idx_adapt_flat = torch.multinomial(weights, num_samples=K_adapt, replacement=True)

                flat_positions = grid.reshape(-1, 3)
                flat_normals = n_grid.reshape(-1, 3)
                verts_adapt = flat_positions.index_select(0, idx_adapt_flat)
                norms_adapt = flat_normals.index_select(0, idx_adapt_flat)

                idx2 = knn_indices(verts_adapt, self.gt_vertices, k=1, chunk=2000)
                matched_normals2 = self.gt_normals[idx2.squeeze(-1)]
                nc_loss_adapt = 1.0 - torch.sum(norms_adapt * matched_normals2, dim=-1).clamp(-1.0, 1.0)
                nc_loss_adapt = nc_loss_adapt.mean()

                nc_loss = 0.5 * nc_loss_uniform + 0.5 * nc_loss_adapt

                batch_source_views = self.renderer.render(vertices, normals, faces, batch_views)

                render_loss = (ref_views.cuda() - batch_source_views).abs().mean()
                
                if rate < 12:
                    w_nc = 0
                else:
                    w_nc = 0.2 * (rate / 16)
                
                
                if boundary_idx_a is not None:
                    boundary_loss = (uniform_vertices.index_select(0, boundary_idx_a) -
                                     uniform_vertices.index_select(0, boundary_idx_b)).abs().mean()
                else:
                    boundary_loss = torch.tensor(0.0, device='cuda')
                
                
                boundary_weight = 0.2 * (rate / 16)
                
                loss = render_loss + laplacian_loss
                loss = render_loss + laplacian_loss + w_nc * nc_loss + boundary_weight * boundary_loss 
                loss = render_loss + laplacian_loss + w_nc * nc_loss 
                loss = render_loss + laplacian_loss + boundary_weight * boundary_loss 


                
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_losses['render'].append(render_loss.item())
                batch_losses['laplacian'].append(laplacian_loss.item())
                batch_losses['boundary'].append(boundary_loss.item())
                batch_losses['normal'].append(nc_loss.item())

            losses['render'].append(np.mean(batch_losses['render']))
            losses['laplacian'].append(np.mean(batch_losses['laplacian']))
            losses['boundary'].append(np.mean(batch_losses['boundary']))           
            losses['normal'].append(np.mean(batch_losses['normal']))
            
            

        logging.info(f'Optimized neural geometry field at resolution ({rate} x {rate})')

        return losses

    def run(self) -> None:
        self.losses = {
            'render': [],
            'laplacian': [],
            'boundary':[],
            'normal': []
        }
        

        self.views = arrange_views(self.target, self.cameras)[0]
        logging.info(f'Generated {self.cameras} views for reference mesh')

        self.reference_views = self.precompute_reference_views()
        logging.info('Cached reference views')

        for rate in [4, 8, 12, 16]:
            opt = torch.optim.Adam(self.nddr.parameters(), 1e-3)
            rate_losses = self.optimize_resolution(opt, rate)
            self.losses['render'] += rate_losses['render']
            self.losses['laplacian'] += rate_losses['laplacian']
            self.losses['normal'] += rate_losses['normal']
            self.losses['boundary'] += rate_losses['boundary']
            # self.display(rate)

        logging.info('Finished training neural geometry field')

    def export(self) -> None:
        # Final export
        self.nddr.save(self.exporter.pytorch())

        logging.info('Exporting neural geometry field as PyTorch (PT)')

        with open(self.exporter.binary(), 'wb') as file:
            file.write(self.nddr.stream())

        logging.info('Exporting neural geometry field as binary')

        # Plot results
        _, axs = plt.subplots(1, 2, layout='constrained')

        axs[0].plot(self.losses['render'], label='Render')
        axs[0].legend()
        axs[0].set_yscale('log')

        axs[1].plot(self.losses['laplacian'], label='Laplacian')
        axs[1].legend()
        axs[1].set_yscale('log')

        plt.savefig(self.exporter.plot("1"))
        
        _, axs = plt.subplots(1, 2, layout='constrained')
        
        axs[0].plot(self.losses['normal'], label='Normal consistency')
        axs[0].legend()
        axs[0].set_yscale('log')
        
        axs[1].plot(self.losses['boundary'], label='Boundary')
        axs[1].legend()
        axs[1].set_yscale('log')
        
        plt.savefig(self.exporter.plot("2"))

        logging.info('Loss history exported')

        # Export mesh
        uvs = self.nddr.sample_uniform(16)
        vertices = self.nddr.eval(*uvs).detach()
        base = self.nddr.base(16).detach()
        cmap = make_cmap(self.nddr.complexes, self.nddr.points.detach(), base, 16)
        remap = nddrutil.generate_remapper(self.nddr.complexes.cpu(), cmap, base.shape[0], 16)
        faces = nddrutil.triangulate_shorted(vertices, self.nddr.complexes.shape[0], 16)
        faces = remap.remap_device(faces)

        mesh = trimesh.Trimesh(vertices=vertices.cpu(), faces=faces.cpu())
        mesh.export(self.exporter.mesh())

        # Write the metadate
        meta = {
            'reference': self.path,
            'torched': os.path.abspath(self.exporter.pytorch()),
            'binaries': os.path.abspath(self.exporter.binary()),
            'stl': os.path.abspath(self.exporter.mesh())
        }

        with open(self.exporter.metadata(), 'w') as file:
            json.dump(meta, file)

    def display(self, rate=16):
        import polyscope as ps

        ps.init()
        ps.register_surface_mesh('Reference',
                                 self.target.vertices.cpu().numpy(),
                                 self.target.faces.cpu().numpy())

        with torch.no_grad():
            base = self.nddr.base(rate).float()
            uvs = self.nddr.sample_uniform(rate)
            vertices = self.nddr.eval(*uvs).float()

        cmap = make_cmap(self.nddr.complexes, self.nddr.points.detach(), base, rate)
        remap = nddrutil.generate_remapper(self.nddr.complexes.cpu(), cmap, base.shape[0], rate)
        faces = nddrutil.triangulate_shorted(vertices, self.nddr.complexes.shape[0], rate)
        faces = remap.remap_device(faces)

        ps.register_surface_mesh('NDDR', vertices.cpu().numpy(), faces.cpu().numpy())
        ps.show()


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',
                        level=logging.INFO,
                        datefmt='%H:%M:%S')

    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh', type=str, help='Target mesh')
    parser.add_argument('--lod', type=int, default=2000, help='Number of patches to partition')
    parser.add_argument('--features', type=int, default=10, help='Feature vector size')
    parser.add_argument('--display', type=bool, default=False, help='Display the result after training')
    parser.add_argument('--batch', type=int, default=10, help='Batch size for training')
    parser.add_argument('--fixed-seed', action='store_true', default=False, help='Fixed random seed (for debugging)')

    args = parser.parse_args()

    if args.fixed_seed:
        torch.manual_seed(0)

    trainer = Trainer(args.mesh, args.lod, args.features, args.batch)
    trainer.run()
    trainer.export()

    if args.display:
        trainer.display()
