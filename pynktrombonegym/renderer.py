from .env import PynkTrombone


class Renderer:
    """Render process of PynkTrombone environment.

    Plotting:
    - current_step
    - current_tract_diameters
    - nose_diameters
    - current voc frequency
    - current voc tenseness
    """

    def __init__(self, env: PynkTrombone, figsize: tuple[float, float] = (6.4, 4.8)) -> None:
        """Construct Renderer and create intial figure and etc...
        Args:
            env (PynkTrombone): PynkTrombone vocal tract environment model.
            figsize (tuple[float, float]): Figure size of rendered image. [Inch]
        """

        self.env = env
        self.figsize = figsize

    def make_infomation_text(self) -> str:
        """Make infomation text displaying on the figure.
        Infomations are:
            - current_step
            - frequency
            - tenseness

        Returns:
            info_text (str): Infomation text value.
        """
        info = (
            f"current step: {self.env.current_step}\n"
            f"frequency: {float(self.env.voc.frequency): .2f}\n"
            f"tenseness: {float(self.env.voc.tenseness): .2f}\n"
        )

        return info
