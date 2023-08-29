import onnx

model_in_file = 'D:\中星微人工智能工作\人车非项目\PytorchOCR_copy\model_weight\\best_rec_pco_model.onnx'

if __name__ == "__main__":
    model = onnx.load(model_in_file)

    nodes = model.graph.node
    nodnum = len(nodes)  # 205

    for nid in range(nodnum):
        if (nodes[nid].output[0] == 'stride_32'):
            print('Found stride_32: index = ', nid)
        else:
            print(nodes[nid].output)

    inits = model.graph.initializer
    ininum = len(inits)  # 124

    for iid in range(ininum):
        el = inits[iid]
        print('name:', el.name, ' dtype:', el.data_type, ' dim:', el.dims)
        # el.raw_data for weights and biases

    print(model.graph.output)  # display all the output nodes

print('Done')