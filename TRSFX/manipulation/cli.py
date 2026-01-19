import click

from .phenix_to_meteor import phenix_to_meteor as func


@click.group()
def cli():
    """
    PSI crystallographic manipulation utilities
    """
    pass


@cli.command()
@click.argument("input", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Output file path. If not provided, appends '_meteor.mtz' to input filename.",
)
def phenix_to_meteor(input, output):
    """
    Command-line interface to convert Phenix-processed MTZ files to Meteor-ready MTZ files.

    Example usage:

    \b
        psi-phenix-to-meteor input_file.mtz --output output_file.mtz
    """
    func(input=input, output=output)


if __name__ == "__main__":
    cli()
