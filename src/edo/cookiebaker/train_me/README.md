# Train Me

## Components

All components are dummy wrappers.

In particular, all components in the pipeline are defined as functions in the library.

```py
@component(base_image=base_image)
def function(a,b,c):
    kwargs = locals()
    from edo.base_module import function
    function(**kwargs)
```

With the component having same args as the function defined inside the library; base_image is a docker image with the py
library installed.

## Debug pipeline
### Local
You can `run_pipeline_local` and enter debug mode in your computer. No need to dockerize, pipeline will run as a py func in your venv.

### Custom google cloud project
You can run pipeline in your own project with `run_pipeline`. Also, the function `pipeline` has the option `DEV`, to just train with a low sample (1 day of data).