from monk.gluon_prototype import prototype

def main(img_name):
    gtf = prototype(verbose=1)
    gtf.Prototype("Task", "gluon_resnet18_v1_train_all_layers", eval_infer=True)
    img_name = "workspace/test/A33344785_I3.jpg"
    predictions = gtf.Infer(img_name=img_name)
    room = predictions['predicted_class']
    print('Done\n')


if __name__ == '__main__':
    print('hi')
    main(img_name)