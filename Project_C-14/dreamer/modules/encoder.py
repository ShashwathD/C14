import torch
import torch.nn as nn

from dreamer.utils.utils import initialize_weights


def _cfg_get(obj, key, default=None):
    if obj is None:
        return default
    try:
        return obj[key]
    except (KeyError, TypeError, AttributeError):
        return getattr(obj, key, default)


class _DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, observation_shape, config):
        super().__init__()
        self.config = config.parameters.dreamer.encoder
        self.observation_shape = observation_shape
        self.embedded_state_size = config.parameters.dreamer.embedded_state_size
        self.flip_hw = bool(_cfg_get(self.config, "flip_hw", False))

        input_channels = int(self.observation_shape[0])
        self.backbone, feature_size = self._build_backbone(input_channels)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Linear(feature_size, self.embedded_state_size)
        self.projection.apply(initialize_weights)

    def _build_backbone(self, input_channels):
        backbone_name = _cfg_get(self.config, "backbone", "mobilenet_v2")
        if backbone_name != "mobilenet_v2":
            return self._build_fallback_backbone(input_channels)

        try:
            from torchvision.models import mobilenet_v2  # type: ignore

            model = mobilenet_v2(weights=None)
            first_conv = model.features[0][0]
            new_conv = nn.Conv2d(
                input_channels,
                first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=False,
            )
            nn.init.kaiming_uniform_(new_conv.weight.data, nonlinearity="relu")
            model.features[0][0] = new_conv
            return model.features, model.last_channel
        except Exception:
            return self._build_fallback_backbone(input_channels)

    def _build_fallback_backbone(self, input_channels):
        network = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            _DepthwiseSeparableBlock(32, 64, stride=1),
            _DepthwiseSeparableBlock(64, 128, stride=2),
            _DepthwiseSeparableBlock(128, 128, stride=1),
            _DepthwiseSeparableBlock(128, 256, stride=2),
            _DepthwiseSeparableBlock(256, 256, stride=1),
            _DepthwiseSeparableBlock(256, 512, stride=2),
            _DepthwiseSeparableBlock(512, 512, stride=1),
        )
        network.apply(initialize_weights)
        return network, 512

    def forward(self, x):
        input_shape = tuple(self.observation_shape)
        batch_with_horizon_shape = x.shape[: -len(input_shape)]
        if not batch_with_horizon_shape:
            batch_with_horizon_shape = (1,)

        x = x.reshape(-1, *input_shape)
        if self.flip_hw:
            # Rotate 180 degrees (equivalent to flipping H and W).
            x = torch.flip(x, dims=[-1, -2])
        x = self.backbone(x)
        x = self.pool(x).flatten(1)
        x = self.projection(x)
        x = x.reshape(*batch_with_horizon_shape, self.embedded_state_size)
        return x
