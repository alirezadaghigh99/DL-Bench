def do_conv2d(conv, input_, *, padding=None, weight=None, bias=None):
    weight = conv.weight if weight is None else weight
    bias = conv.bias if bias is None else bias
    padding = conv.padding if padding is None else padding
    return F.conv2d(input_, weight, bias, conv.stride, padding, conv.dilation, conv.groups)