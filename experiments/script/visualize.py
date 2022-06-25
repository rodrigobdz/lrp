r"""Visualize results from LRP and Pixel-Flipping experiments."""

# pylint: disable=duplicate-code
__author__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__credits__ = ['Rodrigo Bermudez Schettino (TU Berlin)']
__maintainer__ = 'Rodrigo Bermudez Schettino (TU Berlin)'
__email__ = 'r.bermudezschettino@campus.tu-berlin.de'
__status__ = 'Development'
# pylint: enable=duplicate-code

import argparse
from configparser import ConfigParser, ExtendedInterpolation
from pathlib import Path

import numpy
from matplotlib import pyplot as plt
from matplotlib import ticker


def parse_arguments() -> ConfigParser:
    r"""Read configuration file passed as argument.

    :return: ConfigParser object with configuration file contents.
    """
    # Use private variable to avoid pylint warning about redefinition of outside variable.
    _config = ConfigParser(interpolation=ExtendedInterpolation())

    # Define argument parser
    parser = argparse.ArgumentParser(description='Specify the path to the configuration file.',
                                     epilog='For more information,'
                                     ' review the batch_lrp_pf.py script')
    parser.add_argument('-c', '--config-file',
                        type=Path,
                        help='Absolute path to configuration file'
                        ' with parameters for experiments',
                        required=True)
    parsed_args: argparse.Namespace = parser.parse_args()

    # Access parsed arguments
    config_file_path: Path = parsed_args.config_file

    # Ensure that the configuration file exists.
    if not config_file_path.absolute().exists():
        raise ValueError(
            f'Configuration file {config_file_path.absolute()} does not exist.')

    # pylint: disable=pointless-statement
    _config.read(config_file_path)
    # pylint: enable=pointless-statement

    return _config


def contourf_plot():
    r"""Generate contourf plot from experiment results."""
    # Format comments for copy-pasting into Python terminal. Useful for debugging.
    print(f'Importing x, y and z values for plot from files:\n'
          f"x_path = '{PLOT_X_VALUES_PATH}'\n\n"
          f"y_path = '{PLOT_Y_VALUES_PATH}'\n\n"
          f"z_path = '{PLOT_Z_VALUES_PATH}'\n\n"
          f"# Load as: \n\n"
          f"x = numpy.load(x_path, allow_pickle=True)\n\n"
          f"y = numpy.load(y_path, allow_pickle=True)\n\n"
          f"z = numpy.load(z_path, allow_pickle=True)\n\n")

    # Load the values to plot.
    x_values: numpy.ndarray = numpy.load(file=PLOT_X_VALUES_PATH,
                                         allow_pickle=True)
    y_values: numpy.ndarray = numpy.load(file=PLOT_Y_VALUES_PATH,
                                         allow_pickle=True)
    z_values: numpy.ndarray = numpy.load(file=PLOT_Z_VALUES_PATH,
                                         allow_pickle=True)

    # If AUC_SCORE_DECIMALS is not set to -1 (skip rounding), round the values in array.
    if AUC_SCORE_DECIMALS != -1:
        z_values = numpy.round(z_values, decimals=AUC_SCORE_DECIMALS)

    # Dynamically set the frequency of ticks on the z-axis.
    optional_arguments: dict = {}

    print(f'Z values for contourf plot:\n{z_values}\n')

    # Calculate standard deviation of z values to use as measure to prevent excessively
    # fine-grained contourf plots.
    z_std_dev: float = numpy.std(z_values)
    print(f'Standard deviation of Z values: {z_std_dev}\n')

    # In case the standard deviation is too small, treat all AUC scores as equal.
    if z_std_dev <= 0.01:
        print('Standard deviation of Z values is too small.\n'
              'Setting all AUC scores to equal.')
        # Set a value which is close to zero but not zero to avoid error:
        #   "Filled contours require at least 2 levels.".
        optional_arguments = {'locator': ticker.MultipleLocator(base=0.5)}

    # Plot the values.
    plt.tricontourf(x_values,
                    y_values,
                    z_values,
                    **optional_arguments)

    # Set title and axis labels.
    plt.title(TITLE)
    plt.xlabel(X_LABEL)
    plt.ylabel(Y_LABEL)

    # Enable color bar on the side to explain contour levels.
    colorbar: plt.colorbar.Colorbar = plt.colorbar()
    colorbar.ax.set_ylabel('Area Under the Curve (AUC)')

    # Save plot to file.
    plt.savefig(fname=PLOT_PATH, facecolor='w')


if __name__ == "__main__":
    config: ConfigParser = parse_arguments()

    # Path to dir where the results should be stored.
    # Directories will be created, if they don't already exist.
    EXPERIMENT_PARENT_ROOT: str = config['PATHS']['EXPERIMENT_PARENT_ROOT']

    plots_section_name: str = 'PLOTS'
    paths_section_name: str = 'PATHS'

    # Title and axis labels for the plots.
    TITLE: str = config[plots_section_name]['TITLE']
    X_LABEL: str = config[plots_section_name]['X_LABEL']
    Y_LABEL: str = config[plots_section_name]['Y_LABEL']
    AUC_SCORE_DECIMALS: str = config.getint(plots_section_name,
                                            'AUC_SCORE_DECIMALS')

    # Path to save plot to.
    PLOT_PATH: str = config[paths_section_name]['PLOT_PATH']

    # Paths from where to load the values to plot.
    PLOT_X_VALUES_PATH: str = config[paths_section_name]['PLOT_X_VALUES_PATH']
    PLOT_Y_VALUES_PATH: str = config[paths_section_name]['PLOT_Y_VALUES_PATH']
    PLOT_Z_VALUES_PATH: str = config[paths_section_name]['PLOT_Z_VALUES_PATH']

    contourf_plot()

    print(f'''Plot saved to {PLOT_PATH}.

Done plotting.''')
