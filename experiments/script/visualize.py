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
    # Load the values to plot.
    print(f'Importing x, y and z values for plot from files:\n'
          f'x: {PLOT_X_VALUES_PATH}\n'
          f'y: {PLOT_Y_VALUES_PATH}\n'
          f'z: {PLOT_Z_VALUES_PATH}\n')
    x_values: numpy.ndarray = numpy.load(file=PLOT_X_VALUES_PATH,
                                         allow_pickle=True)
    y_values: numpy.ndarray = numpy.load(file=PLOT_Y_VALUES_PATH,
                                         allow_pickle=True)
    z_values: numpy.ndarray = numpy.load(file=PLOT_Z_VALUES_PATH,
                                         allow_pickle=True)

    # Plot the values.
    plt.tricontourf(x_values, y_values, z_values)

    # Set title and axis labels.
    plt.title(TITLE)
    plt.xlabel(X_LABEL)
    plt.ylabel(Y_LABEL)

    # Enable color bar on the side to explain contour levels.
    plt.colorbar()

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

    # Path to save plot to.
    PLOT_PATH: str = config[paths_section_name]['PLOT_PATH']

    # Paths from where to load the values to plot.
    PLOT_X_VALUES_PATH: str = config[paths_section_name]['PLOT_X_VALUES_PATH']
    PLOT_Y_VALUES_PATH: str = config[paths_section_name]['PLOT_Y_VALUES_PATH']
    PLOT_Z_VALUES_PATH: str = config[paths_section_name]['PLOT_Z_VALUES_PATH']

    contourf_plot()

    print(f'''Plot saved to {PLOT_PATH}.
Done plotting.''')
