import os

import streamlit as st


def set_session_state():
    if os.name == 'nt':
        st.session_state['developer'] = True

    elif os.getenv('CLOUD_RUN') == 'True':
        st.session_state['developer'] = True

    if os.getenv('ENVIRONMENT', 'dev').lower() == 'dev':
        st.session_state['dev'] = True
    else:
        st.session_state['dev'] = False

    # always reset raw_data_n
    st.session_state['raw_data_n'] = 0

    email = get_email_iap()
    # organization = get_organization(email)

    session_state_ = dict(
        user_config_error=False,
        submit_text_n=0,
        workspace=False,
        email=email,
        # organization=organization
    )

    for k, v in session_state_.items():
        if k not in st.session_state.keys():
            st.session_state[k] = v



def get_email_iap():
    if "email" in st.session_state.keys():
        return st.session_state['email']

    # if dev:
    #     return 'agent@smith.com'
    if os.name == 'nt':
        return 'agent@smith.com'

    from streamlit.web.server.websocket_headers import _get_websocket_headers
    headers = _get_websocket_headers()
    # st.write(headers)
    email = headers['X-Goog-Authenticated-User-Email']
    email = email.split(':')[1]
    return email


def get_environment_suffix():
    environment = os.getenv('ENVIRONMENT', 'DEV')
    environment = environment.lower()
    environment = '' if environment == 'prod' else f"_{environment}"
    return environment


def server_setup():
    set_session_state()
