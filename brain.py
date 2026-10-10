import argparse
import time

import gymnasium
import flappy_bird_gymnasium
import numpy as np

from flybrain import FlyBrain


# --------------------------------------------------
# Game geometry (pixels, from flappy_bird_gymnasium)
# --------------------------------------------------

SCREEN_W = 288
SCREEN_H = 512
BIRD_X = int(SCREEN_W * 0.2)  # left edge of the bird
BIRD_H = 24
PIPE_W = 52
MAX_VEL = 10
PIPE_SPEED = 4  # px/frame the pipes move left

# Frames of lookahead when deciding a pipe has been passed.
SWITCH_FRAMES = 1

# Bird flaps when its bottom edge would sink below (gap bottom - FLAP_MARGIN).
# A flap lifts the bird ~45 px and the gap is 100 px, so the margin is tight.
FLAP_MARGIN = 4

# Ignore flap commands while the bird is rising faster than this (px/frame).
RISING_LOCKOUT = -1

# --------------------------------------------------
# Brain settings
# --------------------------------------------------

# A single loom pulse below ~0.25 leaves DNp01 silent; at 0.3+ it fires a
# 2-spike burst. So the threshold that decides "flap" lives in the brain.
LOOM_AT_LINE = 0.28  # loom amplitude when the bird is exactly on the flap line
LOOM_PX_SCALE = 40.0  # pixels below/above the line per 1.0 of loom amplitude
BRAIN_STEPS_PER_FRAME = 2  # step 1 receives the pulse, step 2 lets DNp01 answer

# Browser view (--web): game frames per second, and pause between episodes.
WEB_FPS = 30
WEB_RESTART_DELAY = 1.5


def make_brain():
    brain = FlyBrain(device="auto")

    # LC4 / LPLC2 detect looming (something big approaching, e.g. the ground).
    # They drive DNp01 (giant fiber, take-off / jump). LC10a -> DNp01 was tested
    # and never makes DNp01 fire, so it is not used.
    loom_cells = brain.cells(["LC4", "LPLC2"])
    dnp01 = brain.cells(["DNp01"])

    return brain, loom_cells, dnp01


# --------------------------------------------------
# Game state
# --------------------------------------------------


def get_game_state(obs):
    pipes = [
        (float(obs[0]) * SCREEN_W, float(obs[1]) * SCREEN_H, float(obs[2]) * SCREEN_H),
        (float(obs[3]) * SCREEN_W, float(obs[4]) * SCREEN_H, float(obs[5]) * SCREEN_H),
        (float(obs[6]) * SCREEN_W, float(obs[7]) * SCREEN_H, float(obs[8]) * SCREEN_H),
    ]

    # obs[9] is the TOP edge of the bird sprite, not its center.
    bird_top = float(obs[9]) * SCREEN_H
    velocity = float(obs[10]) * MAX_VEL

    # First pipe that will still overlap the bird after the next few frames.
    # Switching to the following gap early stops the bird flapping for a gap
    # it is about to leave (which floats it into the next pipe's top).
    ahead = [p for p in pipes if p[0] + PIPE_W - PIPE_SPEED * SWITCH_FRAMES > BIRD_X]
    pipe_x, gap_top, gap_bottom = min(ahead, key=lambda p: p[0]) if ahead else pipes[-1]

    return {
        "bird_top": bird_top,
        "velocity": velocity,
        "gap_top": gap_top,
        "gap_bottom": gap_bottom,
        "pipe_distance": pipe_x - BIRD_X,
        "pipes": pipes,
    }


def loom_strength(state):
    """How strongly the bird 'sees the ground coming up' next frame.

    Uses where the bird will be after one more frame of gravity, so the
    command is already in flight when the bird reaches the flap line.
    """
    velocity = min(state["velocity"] + 1, MAX_VEL)
    predicted_bottom = state["bird_top"] + velocity + BIRD_H

    flap_line = state["gap_bottom"] - FLAP_MARGIN
    excess = predicted_bottom - flap_line

    return float(np.clip(LOOM_AT_LINE + excess / LOOM_PX_SCALE, 0.0, 1.0))


def render_state(env):
    """What the pygame renderer would draw this frame, so the browser can redraw it with the same sprites."""
    game = env.unwrapped
    return {
        "player": [float(game._player_x), float(game._player_y), float(game._player_rot), int(game._player_idx)],
        "base": [float(game._ground["x"]), float(game._ground["y"])],
        "upper": [[float(p["x"]), float(p["y"])] for p in game._upper_pipes],
        "lower": [[float(p["x"]), float(p["y"])] for p in game._lower_pipes],
    }


# --------------------------------------------------
# Brain -> action
# --------------------------------------------------


def fly_action(obs, brain, loom_cells, dnp01, verbose=False, telemetry=None):
    """telemetry: optional dict, filled with what the brain and game looked like this frame."""
    state = get_game_state(obs)
    loom = loom_strength(state)

    dnp01_spikes = 0
    fired_steps = []
    dn_v, dn_spiked = [], []  # per brain step, one value per DNp01
    started = time.perf_counter()

    for i in range(BRAIN_STEPS_PER_FRAME):
        inject = [(loom_cells, loom)] if (i == 0 and loom > 0) else []
        fired = brain.step(inject=inject)
        fired_steps.append(fired)
        spiked = np.isin(dnp01, fired)
        dnp01_spikes += int(spiked.sum())
        if telemetry is not None:
            # Membrane potential after the step: a neuron that crossed 1.0 has already been reset to 0.
            v = brain.v[dnp01, 0]
            dn_v.append([round(float(x), 4) for x in (v.get() if hasattr(v, "get") else v)])
            dn_spiked.append(spiked.astype(int).tolist())

    brain_ms = (time.perf_counter() - started) * 1000

    if verbose:
        print(
            f"bird_top={state['bird_top']:6.1f}",
            f"vel={state['velocity']:+5.1f}",
            f"gap=({state['gap_top']:.0f},{state['gap_bottom']:.0f})",
            f"loom={loom:.2f}",
            f"DNp01 spikes={dnp01_spikes}",
        )

    # The network is noisy: DNp01 sometimes fires by itself. A flap while the
    # bird is still climbing would throw it into the top pipe, so the wing
    # motor ignores commands until the last flap has run out (efference copy).
    locked = state["velocity"] < RISING_LOCKOUT

    # Giant-fiber spikes = take-off = flap
    action = 0 if locked else int(dnp01_spikes > 0)

    if telemetry is not None:
        telemetry.update(
            fired=np.unique(np.concatenate(fired_steps)),
            loom=loom,
            dn=dnp01_spikes,
            flap=action,
            locked=locked,
            brain_ms=brain_ms,
            bird_top=state["bird_top"],
            velocity=state["velocity"],
            gap=[state["gap_top"], state["gap_bottom"]],
            pipes=state["pipes"],
            dn_v=dn_v,
            dn_spiked=dn_spiked,
        )

    return action


def play(
    render=True,
    seed=None,
    max_frames=None,
    verbose=False,
    brain_parts=None,
    monitor=None,
    fps=None,
    episode=1,
    best=0,
):
    """monitor: a webview.Monitor to stream every frame to. fps: cap the frame rate."""
    env = gymnasium.make(
        "FlappyBird-v0",
        render_mode="human" if render else None,
        use_lidar=False,
        normalize_obs=True,
    )

    brain, loom_cells, dnp01 = brain_parts or make_brain()

    obs, info = env.reset(seed=seed)
    brain.reset(1)

    frames = 0
    score = 0
    telemetry = {} if monitor is not None else None

    while True:
        frame_start = time.perf_counter()
        action = fly_action(obs, brain, loom_cells, dnp01, verbose=verbose, telemetry=telemetry)
        obs, reward, terminated, truncated, info = env.step(action)

        frames += 1
        score = info.get("score", 0)
        done = terminated or truncated or bool(max_frames and frames >= max_frames)

        if monitor is not None:
            monitor.publish(
                telemetry.pop("fired"),
                frame=frames,
                episode=episode,
                score=score,
                best=max(best, score),
                over=done,
                render=render_state(env),
                **telemetry,
            )

        if done:
            break

        if fps:
            time.sleep(max(0.0, 1 / fps - (time.perf_counter() - frame_start)))

    env.close()
    return score, frames


def main():
    parser = argparse.ArgumentParser(description="Flappy Bird played by the fruit-fly connectome.")
    parser.add_argument("--headless", action="store_true", help="do not open the game window")
    parser.add_argument("--verbose", action="store_true", help="print the brain state every frame")
    parser.add_argument("--web", action="store_true", help="live brain-activity page in the browser; plays until Ctrl+C")
    parser.add_argument("--port", type=int, default=8765, help="port for --web (default 8765)")
    parser.add_argument("--no-open", action="store_true", help="with --web, do not open the browser")
    args = parser.parse_args()

    brain_parts = make_brain()
    print("DNp01 neurons:", len(brain_parts[2]))

    if not args.web:
        score, frames = play(render=not args.headless, verbose=args.verbose, brain_parts=brain_parts)
        print("GAME OVER")
        print("Score:", score, "frames:", frames)
        return

    from webview import Monitor

    brain, loom_cells, dnp01 = brain_parts
    monitor = Monitor(
        brain,
        loom_cells,
        dnp01,
        settings={
            "loom_at_line": LOOM_AT_LINE,
            "steps_per_frame": BRAIN_STEPS_PER_FRAME,
            "flap_margin": FLAP_MARGIN,
            "bird_x": BIRD_X,
            "bird_h": BIRD_H,
            "screen_w": SCREEN_W,
            "screen_h": SCREEN_H,
            "pipe_w": PIPE_W,
            # LIF units: spike at v >= 1, reset to 0. With only the tonic drive, v settles
            # where the leak balances it: tonic / (1 - decay).
            "v_threshold": 1.0,
            "v_rest": float(brain.tonic / (1 - brain.decay)),
        },
        port=args.port,
        open_browser=not args.no_open,
    )

    best = 0
    episode = 0
    try:
        while True:
            episode += 1
            score, frames = play(
                render=not args.headless,
                verbose=args.verbose,
                brain_parts=brain_parts,
                monitor=monitor,
                fps=WEB_FPS,
                episode=episode,
                best=best,
            )
            best = max(best, score)
            print(f"Episode {episode}: score {score}, frames {frames} (best {best})")
            time.sleep(WEB_RESTART_DELAY)
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        monitor.close()


if __name__ == "__main__":
    main()
