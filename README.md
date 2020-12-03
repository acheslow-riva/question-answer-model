# transformer
Host an inf1 compiled transformer on AWS. Accessible via Flask API.

# FARM Haystack
The question/answer model is wrapped in a framework that uses the python packages Haystack and FARM, which handles converting logits to readable output, batching inferences, and more.  However, lines 667-673 of the source file FARM/modeling/language_model.py need to be edited for a compiled model to run correctly on an AWS inf1 instance.  Easiest way to do this is to pip install the FARM package from its git repo and edit it there. 
