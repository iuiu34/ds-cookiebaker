"""Main module."""
import os
import re
from importlib.resources import files

import fire
import pandas as pd
from edo.mkt.ml.llm import Messages, LlmBaseModel
from edo.mkt.ml.llm.autogpt import AutoGPT
from edo.mkt.ml.llm.get_secrets import get_secret
from joblib import Parallel, delayed


from edo import cookiebaker
from edo.cookiebaker.train_me.llm_tools import code_reader

os.environ['OPENAI_API_KEY'] = get_secret('OPENAI_API_KEY')


class LlmModel(LlmBaseModel):
    def __init__(self, package_name="prime_renewal", st_empty=None):
        path = files(cookiebaker)
        prompt_ = path.joinpath('model_configuration', 'prompts')
        filename = prompt_.joinpath('prompt.tmpl')
        with open(filename) as f:
            prompt_template = f.read()

        filename = prompt_.joinpath('system.tmpl')
        with open(filename) as f:
            system = f.read()

        system = system.format(
            package_name=package_name
        )

        super().__init__(
            system=system,
            prompt_template=prompt_template,
            # tools=[code_reader],
            # tool_choice="code_reader",
            # st_empty=st_empty,
        )


    def predict_sample(self, prompt):
        # prompt = self.encode_sensitive(prompt)
        # messages = Messages()
        # messages.add_tool_response(booking_id, 'get_booking_id')
        # messages.add_tool_response(prompt, 'get_input_email')
        p = super().predict_sample(prompt=prompt)
        return p.last_content()


def llm_model():
    model = LlmModel()
    path = edo.ai_mail.train_me.llm_model.__name__
    model.app(path)


def main():
    """Execute main program."""
    fire.Fire(llm_model)
    print('\x1b[6;30;42m', 'Success!', '\x1b[0m')


if __name__ == "__main__":
    main()
