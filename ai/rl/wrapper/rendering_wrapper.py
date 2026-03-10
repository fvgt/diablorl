from display import get_dungeon_as_array, dungeon_array_to_image, display_diablo_state
import curses
import collections
import gymnasium as gym


class EventsQueue:
    queue = None
    # Use Braille patterns for representing progress,
    # see here: https://www.unicode.org/charts/nameslist/c_2800.html
    progress = [0x2826, 0x2816, 0x2832, 0x2834]
    progress_cnt = 0

    def __init__(self):
        self.queue = collections.deque(maxlen=10)


class RenderWrapper(gym.Wrapper):
    def __init__(self, env, game_instance, view_radius: int = 10):
        super().__init__(env)
        self.game_instance = game_instance
        self.view_radius = view_radius
        self.events = EventsQueue()

    def get_frame(self):
        d = self.game_instance.safe_state
        arr = get_dungeon_as_array(
            game=self.game_instance, view_radius=self.view_radius
        )
        img = dungeon_array_to_image(arr, scale=10)
        # events_str, _ = get_events_as_string(self.game_instance, self.events)
        # total_hp = diablo_state.count_active_monsters_total_hp(d)
        # draw = ImageDraw.Draw(img)
        # font = ImageFont.truetype("DejaVuSansMono.ttf", size=12)

        # Overlay the info on the image
        # y_offset = 5  # starting y position
        # line_spacing = 12  # space between lines
        #
        # draw.text((5, y_offset), f"Events: {events_str}", fill=(255, 0, 0), font=font)
        # y_offset += line_spacing
        # draw.text((5, y_offset), f"Total Monster HP: {total_hp}", fill=(255, 0, 0), font=font)

        return img  # No need to return event_queue

    def render(self, mode="terminal"):
        if mode == "image":
            return self.get_frame()

        elif mode == "terminal":

            def _curses_render(stdscr):
                display_diablo_state(
                    game=self.game_instance,
                    stdscr=stdscr,
                    events=self.events,
                    envlog=None,
                    view_radius=self.view_radius,
                )
                stdscr.refresh()
                curses.napms(50)

            curses.wrapper(_curses_render)

        else:
            raise ValueError(f"Unknown render mode: {mode}")
