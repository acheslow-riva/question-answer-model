# transformer
Host an inf1 compiled transformer on AWS. Accessible via Flask API.

# FARM Haystack
The question/answer model is wrapped in a framework that uses the python packages Haystack and FARM, which handles converting logits to readable output, batching inferences, and more.  However, lines 667-673 of the source file FARM/modeling/language_model.py need to be edited for a compiled model to run correctly on an AWS inf1 instance.  Easiest way to do this is to pip install the FARM package from its git repo and edit it there. 


# Known issues

## /dev/neuron0 device not found
Sometimes /dev/neuron0 will disappear on the ml-dev-02 server. This may be caused by server upgrades. In order to bring it back, you need to re-install its driver by running these commands:

```bash
sudo -i
cd /opt/
rm -rf aws-neuron-driver
git clone https://github.com/aws/aws-neuron-driver.git
cd aws-neuron-driver/
apt-get update
apt install -y dkms
apt install build-essential linux-source
make
insmod neuron.ko
echo "neuron" | tee -a /etc/modules-load.d/neuron.conf
echo 'KERNEL=="neuron*", MODE="0666"' > /lib/udev/rules.d/neuron-udev.rules
ls dev/neuron*
lsmod |grep -i neuron
```

## RuntimeError
You may encounter the following error: `RuntimeError: NeuronDevice::process_nrtd_response: Context=NRTD_CTX_INITIALIZE NRTD response code 'NERR_HW_ERROR' (6) There was a hardware error. Please contact AWS support. Raw runtime error message ''`

This is caused by a hardware conflict between the host and container. You can fix it by running the following from your sudo enable profile:
```bash
sudo service neuron-rtd stop
```

Then you can restart the containers by running `docker-compose down && docker-compose up -d` from this repo in the `web.mgmt` profile.

## Compile error

```bash
>>> model = torch.neuron.trace(app.finder.reader.inferencer.model.language_model.model, example_inputs=inputs, strict=False)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/usr/local/lib64/python3.7/site-packages/torch_neuron/convert.py", line 71, in trace
    neuron_graph, jit_trace = to_graph(func, example_inputs, return_trace=True)
  File "/usr/local/lib64/python3.7/site-packages/torch_neuron/convert.py", line 228, in to_graph
    jit_trace = torch.jit.trace(func_or_mod, example_inputs)
  File "/usr/local/lib64/python3.7/site-packages/torch/jit/_trace.py", line 742, in trace
    _module_class,
  File "/usr/local/lib64/python3.7/site-packages/torch/jit/_trace.py", line 940, in trace_module
    _force_outplace,
RuntimeError: Encountering a dict at the output of the tracer might cause the trace to be incorrect, this is only valid if the container structure does not change based on the module's inputs. Consider using a constant container instead (e.g. for `list`, use a `tuple` instead. for `dict`, use a `NamedTuple` instead). If you absolutely need this and know the side effects, pass strict=False to trace() to allow this behavior.
```