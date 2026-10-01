"""
Models.  The client model follows the 8 components of the PDF (Part 2, p. 15):

    1 Preprocessor      (components.py)  no parameters  - resample / crop / units per encoder
    2 InputAdapter      (components.py)  local          - 1x1 conv mapping device channels -> encoder channels
    3 Encoder           (encoders/)      frozen         - LIMU-BERT, ssl-wearables, UniMTS, HALO
    4 LoRA modules      (lora.py)        federated      - low-rank updates W = W0 + (alpha/r) B A
    5 Local norm        (inside encoder) local          - LayerNorm / BatchNorm scale+shift
    6 SharedHead g      (components.py)  federated      - features -> 64-d activity embedding
    7 PersonalHead h    (components.py)  local          - 64-d embedding -> class scores
    8 Class prototypes  (components.py)  statistic      - mean embedding per class

`client_model.ClientModel` assembles them and tags every tensor with a ROLE
("encoder", "lora", "norm", "norm_stats", "adapter", "shared_head", "head").
The federated methods (fedpeft/federated/methods.py) then decide per role whether
it is trained, sent to the server, or kept on the device.
"""
