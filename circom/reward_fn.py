import torch
from torch import nn
import random
import matplotlib.pyplot as plt
import json
import ezkl
from typing import Dict, List, Tuple
import bittensor as bt
import os
import numpy as np
import sys

MAX_RESPONSE_TIME = 30
MIN_RESPONSE_TIME = 0
NUM_KEYS_TO_SIMULATE = 1024
# PLOT_INTERVALS = 100
PLOT_INTERVALS = 1
MAX_SCORE = 1 / 256
BATCH_SIZE = 256
ENABLE_LOGS = False
FIX_TIMES_AFTER_INTERVAL = False

# if not os.path.exists("model.compiled"):
#     os.chdir("deployment_layer/pow_model")


class Reward(nn.Module):
    """
    This module is responsible for calculating the reward for a miner based on the provided score, verification_result,
    response_time, and proof_size in its forward pass.
    """

    def __init__(self):
        super().__init__()
        self.RATE_OF_DECAY = torch.tensor(0.4)
        self.RATE_OF_RECOVERY = torch.tensor(0.1)
        self.FLATTENING_COEFFICIENT = torch.tensor(0.9)
        self.PROOF_SIZE_THRESHOLD = torch.tensor(3648)
        self.PROOF_SIZE_WEIGHT = torch.tensor(0)
        self.RESPONSE_TIME_WEIGHT = torch.tensor(1)
        self.MAXIMUM_RESPONSE_TIME_DECIMAL = torch.tensor(0.99)

    def shifted_tan(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tan(
            torch.mul(
                torch.mul(torch.sub(x, torch.tensor(0.5)), torch.pi),
                self.FLATTENING_COEFFICIENT,
            )
        )

    def tan_shift_difference(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sub(self.shifted_tan(x), self.shifted_tan(torch.tensor(0.0)))

    def normalized_tangent_curve(self, x: torch.Tensor) -> torch.Tensor:
        return torch.div(
            self.tan_shift_difference(x), self.tan_shift_difference(torch.tensor(1.0))
        )

    def forward(
        self,
        maximum_score: torch.Tensor,
        previous_score: torch.Tensor,
        verified: torch.Tensor,
        proof_size: torch.Tensor,
        response_time: torch.Tensor,
        maximum_response_time: torch.Tensor,
        minimum_response_time: torch.Tensor,
        block_number: torch.Tensor,
        validator_uid: torch.Tensor,
        miner_uid: torch.Tensor,
    ) -> List[torch.Tensor]:
        rate_of_change = torch.where(
            verified, self.RATE_OF_RECOVERY, self.RATE_OF_DECAY
        )

        response_time_normalized = torch.clamp(
            torch.div(
                torch.sub(response_time, minimum_response_time),
                torch.sub(maximum_response_time, minimum_response_time),
            ),
            0,
            self.MAXIMUM_RESPONSE_TIME_DECIMAL,
        )

        response_time_reward_metric = torch.mul(
            self.RESPONSE_TIME_WEIGHT,
            torch.sub(
                torch.tensor(1), self.normalized_tangent_curve(response_time_normalized)
            ),
        )
        proof_size_reward_metric = torch.mul(
            self.PROOF_SIZE_WEIGHT,
            torch.clamp(
                proof_size / self.PROOF_SIZE_THRESHOLD, torch.tensor(0), torch.tensor(1)
            ),
        )

        calculated_score_fraction = torch.clamp(
            torch.sub(response_time_reward_metric, proof_size_reward_metric),
            torch.tensor(0),
            torch.tensor(1),
        )

        maximum_score = torch.mul(maximum_score, calculated_score_fraction)

        distance_from_score = torch.where(
            verified, torch.sub(maximum_score, previous_score), previous_score
        )

        change_in_score = torch.mul(rate_of_change, distance_from_score)

        new_score = torch.where(
            verified,
            previous_score + change_in_score,
            previous_score - change_in_score,
        )

        return [new_score, block_number, miner_uid, validator_uid]


def log(message: str) -> None:
    if ENABLE_LOGS:
        print(message)


def run_inference_via_proof_system(
    batch_inputs: Dict[str, torch.Tensor]
) -> torch.Tensor:
    log(f"Input batch shape: {batch_inputs['maximum_score'].shape}")
    log(
        f"Sample input values: {batch_inputs['maximum_score'][0]}, {batch_inputs['previous_score'][0]}, {batch_inputs['verified'][0]}"
    )

    with open("input.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "input_data": [batch_inputs[key].tolist() for key in batch_inputs],
                "input_shapes": [list(batch_inputs[key].shape) for key in batch_inputs],
            },
            f,
        )

    try:
        ezkl.gen_witness(
            data="input.json", model="model.compiled", output="witness.json"
        )
        log("Generated witness")
    except Exception as e:
        log(f"Error generating witness: {e}")
        return torch.zeros(BATCH_SIZE, dtype=torch.float64)

    with open("settings.json", "r", encoding="utf-8") as settings_file:
        settings_json = json.load(settings_file)
        scaling_factor = settings_json["model_output_scales"][0]

    with open("witness.json", "r", encoding="utf-8") as witness_file:
        witness_json = json.load(witness_file)
        output_values = witness_json["outputs"]
        try:
            hex_tensor = torch.tensor(
                [ezkl.felt_to_int(val) for val in output_values[0]], dtype=torch.float64
            )
            scale_tensor = torch.tensor([2.0**scaling_factor], dtype=torch.float64)
            result = torch.div(hex_tensor, scale_tensor)
            log(f"Processed output sample: {result[:5]}...")
            return result
        except ValueError as e:
            log(f"Error converting output values: {e}")
            return torch.zeros(BATCH_SIZE, dtype=torch.float64)


def generate_sample_inputs() -> Dict[str, torch.Tensor]:
    inputs = {
        "maximum_score": torch.full((BATCH_SIZE,), MAX_SCORE, dtype=torch.float32),
        "previous_score": torch.tensor(
            [random.uniform(0, MAX_SCORE) for _ in range(BATCH_SIZE)],
            dtype=torch.float32,
        ),
        "verified": torch.tensor(
            [i != BATCH_SIZE - 1 for i in range(BATCH_SIZE)], dtype=torch.bool
        ),
        "proof_size": torch.tensor(
            [random.randint(0, 10000) for _ in range(BATCH_SIZE)], dtype=torch.int32
        ),
        "response_time": torch.tensor(
            [
                random.uniform(MIN_RESPONSE_TIME, MAX_RESPONSE_TIME + 2)
                for _ in range(BATCH_SIZE)
            ],
            dtype=torch.float32,
        ),
        "maximum_response_time": torch.full(
            (BATCH_SIZE,), MAX_RESPONSE_TIME, dtype=torch.float32
        ),
        "minimum_response_time": torch.full(
            (BATCH_SIZE,), MIN_RESPONSE_TIME, dtype=torch.float32
        ),
        "block_number": torch.tensor(
            [random.randint(3000000, 10000000) for _ in range(BATCH_SIZE)],
            dtype=torch.int32,
        ),
        "validator_uid": torch.tensor(
            [random.randint(0, 256) for _ in range(BATCH_SIZE)], dtype=torch.int32
        ),
        "miner_uid": torch.tensor(
            [random.randint(0, 256) for _ in range(BATCH_SIZE)], dtype=torch.int32
        ),
    }
    log(
        f"Generated sample inputs. Shape of 'maximum_score': {inputs['maximum_score'].shape}"
    )
    return inputs


def run_iterations(
    num_iterations: int, initial_inputs: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sample_inputs = initial_inputs.copy()
    original_response_times = sample_inputs["response_time"].clone()
    original_verified = sample_inputs["verified"].clone()

    scores_pow = [sample_inputs["previous_score"].clone()]
    scores_torch = [sample_inputs["previous_score"].clone()]
    reward_model = Reward()

    for i in range(num_iterations):
        log(f"Running iteration {i+1}/{num_iterations}")
        if FIX_TIMES_AFTER_INTERVAL:
            if i == num_iterations // 3:
                sample_inputs["response_time"] = torch.full(
                    (BATCH_SIZE,), 4.0, dtype=torch.float32
                )
            elif i == 2 * num_iterations // 3:
                sample_inputs["response_time"] = torch.full(
                    (BATCH_SIZE,), 10.0, dtype=torch.float32
                )

        new_scores_pow = run_inference_via_proof_system(sample_inputs)
        new_scores_torch = reward_model(**sample_inputs)[0]
        sys.exit()

        scores_pow.append(new_scores_pow)
        scores_torch.append(new_scores_torch)

        sample_inputs["previous_score"] = new_scores_pow.clone()
        log(
            f"Updated previous_score for next iteration: {sample_inputs['previous_score'][:5]}..."
        )

    log(
        f"Completed {num_iterations} iterations. Final scores shape: {scores_pow[-1].shape}"
    )
    return (
        torch.stack(scores_pow),
        torch.stack(scores_torch),
        original_response_times,
        original_verified,
    )


def plot_scores_over_time(
    scores_over_time_pow: torch.Tensor,
    scores_over_time_torch: torch.Tensor,
    response_times: torch.Tensor,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

    norm = plt.Normalize(vmin=MIN_RESPONSE_TIME, vmax=MAX_RESPONSE_TIME)
    cmap = plt.get_cmap("plasma")

    for i in range(BATCH_SIZE):
        color = cmap(norm(response_times[i].item()))
        ax1.plot(
            range(PLOT_INTERVALS + 1),
            scores_over_time_pow[:, i],
            color=color,
            alpha=0.5,
        )
    sm1 = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm1.set_array([])
    fig.colorbar(sm1, ax=ax1, label="Response Time")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Score")
    ax1.set_title("Change in Scores Over Iterations (PoW)")

    for i in range(BATCH_SIZE):
        color = cmap(norm(response_times[i].item()))
        ax2.plot(
            range(PLOT_INTERVALS + 1),
            scores_over_time_torch[:, i],
            color=color,
            alpha=0.5,
        )
    sm2 = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm2.set_array([])
    fig.colorbar(sm2, ax=ax2, label="Response Time")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Score")
    ax2.set_title("Change in Scores Over Iterations (Torch)")

    plt.tight_layout()
    plt.show()


def plot_scores_vs_response_times_with_loss(
    final_scores_pow: torch.Tensor,
    final_scores_torch: torch.Tensor,
    response_times: torch.Tensor,
    verified: torch.Tensor,
    loss: torch.Tensor,
) -> None:
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))

    sorted_indices = torch.argsort(response_times, descending=True)
    sorted_response_times = response_times[sorted_indices]
    sorted_scores_pow = final_scores_pow[sorted_indices]
    sorted_scores_torch = final_scores_torch[sorted_indices]
    sorted_verified = verified[sorted_indices]

    scatter_pow = ax1.scatter(
        sorted_response_times[sorted_verified],
        sorted_scores_pow[sorted_verified],
        c=sorted_response_times[sorted_verified],
        cmap="plasma",
        label="Verified (PoW)",
    )
    ax1.scatter(
        sorted_response_times[~sorted_verified],
        sorted_scores_pow[~sorted_verified],
        c="red",
        label="Non-Verified (PoW)",
    )
    fig.colorbar(scatter_pow, ax=ax1, label="Response Time")
    ax1.set_xlabel("Response Time")
    ax1.set_ylabel("Final Score")
    ax1.set_title("Final Scores vs Response Times (PoW)")
    ax1.legend()
    ax1.invert_xaxis()

    scatter_torch = ax2.scatter(
        sorted_response_times[sorted_verified],
        sorted_scores_torch[sorted_verified],
        c=sorted_response_times[sorted_verified],
        cmap="plasma",
        label="Verified (Torch)",
    )
    ax2.scatter(
        sorted_response_times[~sorted_verified],
        sorted_scores_torch[~sorted_verified],
        c="red",
        label="Non-Verified (Torch)",
    )
    fig.colorbar(scatter_torch, ax=ax2, label="Response Time")
    ax2.set_xlabel("Response Time")
    ax2.set_ylabel("Final Score")
    ax2.set_title("Final Scores vs Response Times (Torch)")
    ax2.legend()
    ax2.invert_xaxis()

    ax3.plot(range(len(loss)), loss.numpy())
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("Mean Squared Error")
    ax3.set_title("Loss between Torch and PoW")

    plt.tight_layout()
    plt.show()

    log(f"PoW - Sorted final scores sample: {sorted_scores_pow[:5]}...")
    log(f"PoW - Sorted response times sample: {sorted_response_times[:5]}...")
    log(f"PoW - Sorted verified status sample: {sorted_verified[:5]}...")
    log(f"Torch - Sorted final scores sample: {sorted_scores_torch[:5]}...")
    log(f"Torch - Sorted response times sample: {sorted_response_times[:5]}...")
    log(f"Torch - Sorted verified status sample: {sorted_verified[:5]}...")


def process_weights(
    scores: torch.Tensor, uids: torch.Tensor, netuid: int, subtensor: bt.subtensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    return bt.utils.weight_utils.process_weights_for_netuid(
        uids=uids, weights=scores, netuid=netuid, subtensor=subtensor
    )


def plot_processed_weights(
    processed_uids: torch.Tensor,
    processed_weights: torch.Tensor,
    response_times: torch.Tensor,
    verified: torch.Tensor,
    title: str,
) -> None:
    sorted_indices = torch.argsort(response_times, descending=True)
    sorted_uids = processed_uids[sorted_indices]
    sorted_weights = processed_weights[sorted_indices]
    sorted_response_times = response_times[sorted_indices]
    sorted_verified = verified[sorted_indices]

    plt.figure(figsize=(12, 6))
    scatter = plt.scatter(
        sorted_response_times.numpy(),
        sorted_weights.numpy(),
        c=sorted_response_times.numpy(),
        cmap="plasma",
        s=50,
        alpha=0.7,
    )

    plt.scatter(
        sorted_response_times[~sorted_verified].numpy(),
        sorted_weights[~sorted_verified].numpy(),
        facecolors="none",
        edgecolors="red",
        s=100,
        linewidths=2,
    )

    plt.colorbar(scatter, label="Response Time")
    plt.xlabel("Response Time")
    plt.ylabel("Processed Weight")
    plt.title(title)
    plt.gca().invert_xaxis()
    plt.show()


def plot_processed_weights_grid(
    processed_uids: torch.Tensor,
    processed_weights: torch.Tensor,
    response_times: torch.Tensor,
    verified: torch.Tensor,
    title: str,
) -> None:
    num_charts = BATCH_SIZE // 256
    rows = int(np.ceil(np.sqrt(num_charts)))
    cols = int(np.ceil(num_charts / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    fig.suptitle(title, fontsize=16)

    for i in range(num_charts):
        ax = axes[i // cols, i % cols] if num_charts > 1 else axes
        start_idx = i * 256
        end_idx = (i + 1) * 256

        chunk_uids = processed_uids[start_idx:end_idx]
        chunk_weights = processed_weights[start_idx:end_idx]
        chunk_response_times = response_times[start_idx:end_idx]
        chunk_verified = verified[start_idx:end_idx]
        sorted_indices = torch.argsort(chunk_response_times, descending=True)
        sorted_uids = chunk_uids[sorted_indices]
        sorted_weights = chunk_weights[sorted_indices]
        sorted_response_times = chunk_response_times[sorted_indices]
        sorted_verified = chunk_verified[sorted_indices]

        scatter = ax.scatter(
            sorted_response_times.numpy(),
            sorted_weights.numpy(),
            c=sorted_response_times.numpy(),
            cmap="plasma",
            s=20,
            alpha=0.7,
        )

        ax.scatter(
            sorted_response_times[~sorted_verified].numpy(),
            sorted_weights[~sorted_verified].numpy(),
            facecolors="none",
            edgecolors="red",
            s=40,
            linewidths=1,
        )

        ax.set_xlabel("Response Time")
        ax.set_ylabel("Processed Weight")
        ax.set_title(f"Chunk {i+1}")

        ax.invert_xaxis()

    for i in range(num_charts, rows * cols):
        fig.delaxes(axes.flatten()[i])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def main() -> None:
    initial_inputs = generate_sample_inputs()

    print([initial_inputs[i].shape for i in initial_inputs.keys()])
    # sys.exit()
    scores_over_time_pow, scores_over_time_torch, response_times, verified = (
        run_iterations(PLOT_INTERVALS, initial_inputs)
    )
    plot_scores_over_time(scores_over_time_pow, scores_over_time_torch, response_times)

    loss = torch.mean((scores_over_time_torch - scores_over_time_pow) ** 2, dim=1)

    final_scores_pow = scores_over_time_pow[-1]
    final_scores_torch = scores_over_time_torch[-1]

    sorted_indices = torch.argsort(response_times, descending=True)
    sorted_final_scores_pow = final_scores_pow[sorted_indices]
    sorted_final_scores_torch = final_scores_torch[sorted_indices]
    sorted_response_times = response_times[sorted_indices]
    sorted_verified = verified[sorted_indices]

    plot_scores_vs_response_times_with_loss(
        sorted_final_scores_pow,
        sorted_final_scores_torch,
        sorted_response_times,
        sorted_verified,
        loss,
    )

    uids = torch.arange(len(final_scores_pow))
    netuid = 2
    subtensor = bt.subtensor()
    processed_uids_pow, processed_weights_pow = process_weights(
        final_scores_pow, uids, netuid, subtensor
    )
    plot_processed_weights(
        processed_uids_pow,
        processed_weights_pow,
        response_times,
        verified,
        "Processed Weights (PoW)",
    )
    plot_processed_weights_grid(
        processed_uids_pow,
        processed_weights_pow,
        response_times,
        verified,
        "Processed Weights Grid (PoW)",
    )
    processed_uids_torch, processed_weights_torch = process_weights(
        final_scores_torch, uids, netuid, subtensor
    )
    plot_processed_weights(
        processed_uids_torch,
        processed_weights_torch,
        response_times,
        verified,
        "Processed Weights (Torch)",
    )
    plot_processed_weights_grid(
        processed_uids_torch,
        processed_weights_torch,
        response_times,
        verified,
        "Processed Weights Grid (Torch)",
    )


if __name__ == "__main__":
    main()