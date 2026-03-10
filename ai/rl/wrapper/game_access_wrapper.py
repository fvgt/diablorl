import gymnasium as gym


class EasyGameAccessWrapper(gym.Wrapper):
    def __init__(self, env, game_instance):
        super().__init__(env)
        self.game = game_instance

    def __getattr__(self, name):
        # If the attribute exists on this wrapper, return it
        if name in self.__dict__:
            return self.__dict__[name]
        # Otherwise, delegate to the wrapped env
        return getattr(self.env, name)
