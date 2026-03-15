import numpy as np
from diablo_env import ActionEnum, DiabloEnv
import diablo_state


def build_action_mask_fn(gameconfig, n_actions):

    def get_action_mask(obs):
        # Ensure n_actions matches your ActionEnum size (e.g., 8 for movement)
        masks = np.ones(shape=(obs.shape[0], n_actions), dtype=bool)

        # Map neighbor index to the corresponding Walk Action
        # This assumes get_matrix_neighbors returns: 0:N, 1:NE, 2:E, 3:SE, 4:S, 5:SW, 6:W, 7:NW
        idx_to_action = {
            0: ActionEnum.Walk_N.value,
            1: ActionEnum.Walk_NE.value,
            2: ActionEnum.Walk_E.value,
            3: ActionEnum.Walk_SE.value,
            4: ActionEnum.Walk_S.value,
            5: ActionEnum.Walk_SW.value,
            6: ActionEnum.Walk_W.value,
            7: ActionEnum.Walk_NW.value,
        }

        for i, o in enumerate(obs):
            v_radius = gameconfig["view-radius"]
            player_pos = (v_radius, v_radius)

            # neighbors are [N -> clockwise to NE]
            assert (
                o[player_pos[0], player_pos[1]]
                & diablo_state.EnvironmentFlag.Player.value
            )
            neighbors = DiabloEnv.get_matrix_neighbors(o, *player_pos)
            # print(diablo_state.player_position(d=e.game.safe_state))

            # Check every neighbor index (0 through 7)
            for idx, tile_flag in enumerate(neighbors):
                # If the tile contains a Wall flag
                if tile_flag & diablo_state.EnvironmentFlag.Wall.value:
                    # Get the action associated with this neighbor index
                    forbidden_action = idx_to_action[idx]
                    # Set mask to False (Forbidden)
                    masks[i, forbidden_action] = False
        # do not allow standing
        masks[:, 8] = False

        return masks

    return get_action_mask
