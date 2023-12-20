import subprocess

import fire
import yaml
from edo.kfp_local.kfp import get_docker_build_vm


def deploy_app(docker: bool = True,
               cloud_run: bool = False,
               app_engine: bool = False,
               dev: bool = False):
    if cloud_run and app_engine:
        raise ValueError('Cannot deploy to both Cloud Run and App Engine. Choose one.')

    if app_engine:
        filename = 'app.yaml'
    else:
        filename = 'service.yaml'

    if dev:
        filename = filename.split('.')
        filename = f"{filename[0]}_dev.{filename[1]}"

    with open(filename) as f:
        app_config = yaml.full_load(f)

    if app_engine:
        image_url = app_config['env_variables']['IMAGE_URL']
    else:
        image_url = app_config['spec']['template']['spec']['containers'][0]['image']

    if docker:
        # cmd = 'container/vm_docker_build.sh'
        # cmd = f'sh {cmd} {image_url}'
        # subprocess.run(cmd)
        get_docker_build_vm(
            image_url, package="sorting_hat_fare_rules_app",
            repo="ds-csi-sorting-hat-fare-rules-app")

    if cloud_run:
        # service_name = app_config['metadata']['name']
        # cmd = f'sh gcloud run services delete {service_name} -q'
        # subprocess.run(cmd)
        cmd = f"sh gcloud run services replace {filename}"
        subprocess.run(cmd, check=True)
        # cmd = f'sh gcloud run services add-iam-policy-binding {service_name} ' \
        #       f'--member="allUsers" --role="roles/run.invoker"'
        # subprocess.run(cmd)
    elif app_engine:
        cmd = f"sh gcloud app deploy {filename} -q --image-url={image_url}"
        subprocess.run(cmd, check=True)
        # webbrowser.open(url)


def main():
    """Execute main program."""
    fire.Fire(deploy_app)
    print('\x1b[6;30;42m', 'Success!', '\x1b[0m')


if __name__ == "__main__":
    main()
