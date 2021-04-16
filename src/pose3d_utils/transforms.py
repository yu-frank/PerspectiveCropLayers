from abc import ABC, abstractmethod

import math

from pose3d_utils import mat4, mat3
from pose3d_utils.transformers import TransformerContext


class Transform(ABC):
    @abstractmethod
    def add_camera_transform(self, ctx: TransformerContext):
        pass

    @abstractmethod
    def add_image_transform(self, ctx: TransformerContext):
        pass

    @abstractmethod
    def add_point_transform(self, ctx: TransformerContext):
        pass


class AdjustColour(Transform):
    def __init__(self, brightness, contrast, saturation, hue):
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def add_camera_transform(self, ctx):
        pass

    def add_image_transform(self, ctx):
        ctx.image_transformer.adjust_colour(
            self.brightness, self.contrast, self.saturation, self.hue)

    def add_point_transform(self, ctx):
        pass


class HorizontalFlip(Transform):
    def __init__(self, flip_indices, do_flip):
        super().__init__()
        self.flip_indices = flip_indices
        self.do_flip = do_flip

    def add_camera_transform(self, ctx):
        pass

    def add_image_transform(self, ctx):
        if self.do_flip:
            ctx.image_transformer.mm(mat3.flip_x())

    def add_point_transform(self, ctx):
        if self.do_flip:
            ctx.point_transformer.mm(mat4.flip_x())
            ctx.point_transformer.reorder_points(self.flip_indices)


class ZoomImage(Transform):
    def __init__(self, sx, sy=None):
        super().__init__()
        self.sx = sx
        self.sy = self.sx if sy is None else sy

    def add_camera_transform(self, ctx):
        ctx.camera_transformer.mm(mat3.stretch(self.sx, self.sy))

    def add_image_transform(self, ctx):
        ctx.image_transformer.mm(mat3.stretch(self.sx, self.sy))

    def add_point_transform(self, ctx):
        pass


class RotateImage(Transform):
    def __init__(self, degrees):
        super().__init__()
        self.radians = math.radians(degrees)

    def add_camera_transform(self, ctx):
        pass

    def add_image_transform(self, ctx):
        ctx.image_transformer.mm(mat3.rotate(self.radians))

    def add_point_transform(self, ctx):
        camera = ctx.camera_transformer.transform(ctx.orig_camera)
        k = camera.alpha_y / camera.alpha_x
        rads = self.radians
        ctx.point_transformer.mm(mat4.affine(
            A=[[ math.cos(rads), math.sin(rads) * k, 0],
               [-math.sin(rads) / k, math.cos(rads), 0],
               [              0,              0, 1]],
        ))


class PanImage(Transform):
    def __init__(self, dx, dy):
        super().__init__()
        self.dx = dx
        self.dy = dy

    def add_camera_transform(self, ctx):
        pass

    def add_image_transform(self, ctx):
        ctx.image_transformer.mm(mat3.translate(self.dx, self.dy))

    def add_point_transform(self, ctx):
        camera = ctx.camera_transformer.transform(ctx.orig_camera)

        ox = self.dx / camera.alpha_x
        oy = self.dy / camera.alpha_y
        ctx.point_transformer.mm(mat4.affine(
            A=[[1, 0, ox],
               [0, 1, oy],
               [0, 0,  1]],
        ))


class ChangeResolution(Transform):
    def __init__(self, out_width, out_height):
        super().__init__()
        self.out_width = out_width
        self.out_height = out_height

    def add_camera_transform(self, ctx):
        in_width, in_height = ctx.image_transformer.output_size
        sx = self.out_width / in_width
        sy = self.out_height / in_height
        ctx.camera_transformer.zoom(sx, sy)

    def add_image_transform(self, ctx):
        old_dest_size = ctx.image_transformer.dest_size
        new_dest_size = ctx.image_transformer.set_output_size(self.out_width, self.out_height)
        scale = new_dest_size / old_dest_size
        ctx.image_transformer.mm(mat3.stretch(scale[0], scale[1]))

    def add_point_transform(self, ctx):
        pass

class SquareCrop(Transform):
    def __init__(self):
        super().__init__()

    def add_camera_transform(self, ctx):
        image_width, image_height = ctx.image_transformer.output_size
        if image_height < image_width:
            sx = image_width / image_height
            sy = 1
        else:
            sx = 1
            sy = image_height / image_width
        ctx.camera_transformer.mm(mat3.stretch(sx, sy))

    def add_image_transform(self, ctx):
        image_width, image_height = ctx.image_transformer.output_size
        if image_height < image_width:
            sx, sy = image_width / image_height, 1
        else:
            sx, sy = 1, image_height / image_width
        ctx.image_transformer.mm(mat3.stretch(sx, sy))

    def add_point_transform(self, ctx):
        pass
