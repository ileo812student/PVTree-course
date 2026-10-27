import torch

from models.networks import define_D, define_E, define_G
from style_generate import isolated_sample_rng, stable_sample_seed, trace_torch_randn_calls


def test_pce_network_forward_shapes():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_ids = [0] if device.type == "cuda" else []

    generator = define_G(
        1,
        1,
        8,
        8,
        netG="unet_128",
        norm="instance",
        gpu_ids=gpu_ids,
        where_add="all",
        upsample="bilinear",
    ).eval()
    encoder = define_E(
        1,
        8,
        8,
        "resnet_128",
        norm="instance",
        gpu_ids=gpu_ids,
        vaeLike=True,
    ).eval()
    discriminator = define_D(
        1,
        8,
        "basic_128_multi",
        norm="instance",
        num_Ds=2,
        gpu_ids=gpu_ids,
    ).eval()

    image = torch.randn(1, 1, 128, 128, device=device)
    latent = torch.randn(1, 8, device=device)
    with torch.no_grad():
        generated = generator(image, latent)
        mean, log_variance = encoder(generated)
        scores = discriminator(generated)

    assert generated.shape == (1, 1, 128, 128)
    assert mean.shape == (1, 8)
    assert log_variance.shape == (1, 8)
    assert len(scores) == 2
    assert all(score.shape[0:2] == (1, 1) for score in scores)


def test_path_noise_seed_is_stable_and_path_specific():
    first = stable_sample_seed(2028, "c1_0/c1_0_sample0.png")
    assert first == stable_sample_seed(2028, "c1_0/c1_0_sample0.png")
    assert first == stable_sample_seed(2028, "c1_0\\c1_0_sample0.png")
    assert first != stable_sample_seed(2028, "c1_0/c1_0_sample1.png")
    assert first != stable_sample_seed(2027, "c1_0/c1_0_sample0.png")


def test_isolated_sample_rng_restores_state_and_repeats():
    torch.manual_seed(77)
    outer_before = torch.get_rng_state().clone()
    with isolated_sample_rng(2028, "c1_0/c1_0_sample0.png", []) as first_evidence:
        first = torch.randn(4, 5)
    assert torch.equal(outer_before, torch.get_rng_state())
    with isolated_sample_rng(2028, "c1_0/c1_0_sample0.png", []) as second_evidence:
        second = torch.randn(4, 5)
    assert torch.equal(first, second)
    assert first_evidence["sample_noise_seed"] == second_evidence["sample_noise_seed"]
    assert first_evidence["cuda_rng_before_sha256"] == []
    assert first_evidence["cuda_rng_after_sha256"] == []


def test_randn_trace_observes_without_changing_values():
    torch.manual_seed(11)
    expected = torch.randn(2, 3)
    torch.manual_seed(11)
    with trace_torch_randn_calls() as calls:
        actual = torch.randn(2, 3)
    assert torch.equal(actual, expected)
    assert len(calls) == 1
    assert calls[0]["shape"] == [2, 3]
    assert len(calls[0]["sha256"]) == 64
