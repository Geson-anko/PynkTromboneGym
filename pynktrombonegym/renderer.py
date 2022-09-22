import matplotlib.pyplot as plt

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
        self.create_initial_components()

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

    def create_initial_components(self) -> None:
        """Create components and set them as attribute.
        Componentes are Figure, axes, lines, and etc...
        """

        nose_diameters = self.env.voc.nose_diameters
        currect_tract_diameters = self.env.voc.current_tract_diameters

        self.figure = plt.figure(figsize=self.figsize)
        self.axes = self.figure.add_subplot(1, 1, 1)

        self.indices = list(range(self.env.voc.tract_size))
        self.nose_indices = self.indices[-self.env.voc.nose_size :]

        self.axes.set_ylim(0.0, 5.0)
        self.nose_diameters_line = self.axes.plot(self.nose_indices, nose_diameters, label="nose diameters")[0]
        self.tract_diameters_line = self.axes.plot(self.indices, currect_tract_diameters, label="tract diameters")[0]
        self.axes.legend()

        self.axes.set_title("Tract diameters")
        self.axes.set_xlabel("diameter index")
        self.axes.set_ylabel("diameter [cm]")

        info = self.make_infomation_text()

        self.infomation_text = self.axes.text(1, 4.0, info)
