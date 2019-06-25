import click
import ast
import logging


LOGGER = logging.getLogger(__name__)


class PythonLiteralOption(click.Option):

    def type_cast_value(self, ctx, value):
        try:
            return ast.literal_eval(value)
        except Exception as e:
            LOGGER.error(f"Malformed click input for PythonLiteralOption {e}", exc_info=True)
            raise click.BadParameter(value)
