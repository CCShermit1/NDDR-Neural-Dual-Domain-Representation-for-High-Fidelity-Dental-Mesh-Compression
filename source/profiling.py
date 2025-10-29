import torch
import nddrutil
import torch.autograd.profiler as profiler

from source.nddr import NDDR
from render import Renderer
from util import *


if __name__ == '__main__':
    reference_path = 'meshes/xyz.stl'
    base_path = 'results/quadrangulated/xyz-lod1000-f20.obj'

    target, normalizer = load_mesh(reference_path)
    renderer = Renderer()
    views = arrange_views(target, 20)[0]
    nddr = NDDR.from_base(base_path, normalizer, 16)

    # Get the reference views
    vertices = target.vertices
    vertices = vertices[target.faces].reshape(-1, 3)
    faces = torch.arange(vertices.shape[0])
    faces = faces.int().cuda().reshape(-1, 3)
    normals = vertex_normals(vertices, faces)

    reference_views = renderer.interpolate(*separate(vertices, faces), views)

    # Laplacian setup
    rate = 16
    base = nddr.base(rate).detach()
    cmap = make_cmap(nddr.complexes, nddr.points.detach(), base, rate)
    remap = nddrutil.generate_remapper(nddr.complexes.cpu(), cmap, base.shape[0], rate)
    quads = torch.from_numpy(quadify(nddr.complexes.shape[0], rate)).int()
    graph = nddrutil.Graph(remap.remap(quads), base.shape[0])

    # Run profiler on an iteration
    optimizer = torch.optim.Adam(nddr.parameters(), 1e-3)
    with profiler.profile(with_stack=True, profile_memory=True) as prof:
        uvs = nddr.sampler(rate)
        vertices = nddr.eval(*uvs)
        smoothed_vertices = graph.smooth(vertices, 1.0)

        faces = nddrutil.triangulate_shorted(vertices, nddr.complexes.shape[0], rate)
        faces = remap.remap_device(faces)
        # normals = vertex_normals(vertices, faces)

        batch_source_views = renderer.interpolate(*separate(vertices, faces), views)

        laplacian_loss = (vertices - smoothed_vertices).abs().mean()
        render_loss = (reference_views.cuda() - batch_source_views).abs().mean()
        loss = laplacian_loss + render_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(prof.key_averages(group_by_stack_n=5).table(sort_by='self_cpu_time_total', row_limit=-1))
    prof.export_chrome_trace('trace.json')
