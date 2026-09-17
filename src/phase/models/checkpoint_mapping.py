"""State-dictionary mappings for externally released baseline checkpoints."""


def map_official_tfno_state_dict(state_dict):
    """Map released PhysicsNeMo-style tFNO keys to PHASE tFNO keys."""
    mapped = {}
    for key, value in state_dict.items():
        if key == "decoder_net.device_buffer":
            continue
        key = key.replace(
            "spec_encoder.lift_network.0.conv.",
            "spec_encoder.lift_network.0.linear.",
        )
        key = key.replace(
            "spec_encoder.lift_network.2.conv.",
            "spec_encoder.lift_network.2.linear.",
        )
        key = key.replace("decoder_net.layers.0.linear.", "decoder_net.net.0.")
        key = key.replace("decoder_net.final_layer.linear.", "decoder_net.net.2.")
        mapped[key] = value
    return mapped
