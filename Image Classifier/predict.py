# predit
if __name__ == "__main__":
    import argparse
    import Common as common
    import importlib
    import time

    importlib.reload(common)
    
    parser = argparse.ArgumentParser(description= 'Welcome to Image Classifier predition part.')
    
    
    # python predict.py /path/to/image checkpoint
    parser.add_argument('input', action='store', default='flowers\images.jpg',
                        help='Enter the image path')
    
    parser.add_argument('checkpoint', action='store', default='checkpoint.pth')
    
    # python predict.py input checkpoint --top_k 3
    parser.add_argument('--top_k', action='store', type=int, default='1')
    # python predict.py input checkpoint --category_names cat_to_name.json
    parser.add_argument('--category_names', action='store', type=str, default='cat_to_name.json')
    # python predict.py input checkpoint --gpu
    parser.add_argument('--gpu', action="store_true", default=False,
                        help="Enable GPU mode. Default off.")
    
    read_line = parser.parse_args()
    input_path = read_line.input
    checkpoint_path = read_line.checkpoint
    category_names_path = read_line.category_names
    top_k = read_line.top_k
    gpu = read_line.gpu
    
    common.init(gpu)
    
    # Start Predict
    start_time = time.time()
    common.load_checkpoint(checkpoint_path)
    image = common.process_image(input_path)
    top_ps, top_class, top_class_name = common.predict(image, top_k, category_names_path)
    if(top_k > 1):
        print(f'Top {top_k} rank')
        [print(f'{idx + 1}: {top_class_name[idx]}, probability is {"%.3f" %top_ps[0][idx]} \n')  for idx in range(top_k)]
    else:
        print(f'Flower name: {top_class_name[0]}, Probability:{"%.3f" %top_ps[0][0]}')
    elapsed_time = time.time() - start_time
    print(f'Total Traning Time: {"%.2f" %elapsed_time}s')
    
    #print(f"Input Path:{input_path}")
    #print(f"Checkpoint Path:{checkpoint_path}")
    #print(f"Category Names:{category_names_path}")
    #print(f"Input Path:{gpu}")