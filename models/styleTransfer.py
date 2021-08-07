# import resources
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.optim as optim
import requests
from torchvision import transforms, models

class styleTransfer():
    def __init__(self):
        #get the vgg19 model
        self.vgg = models.vgg19(pretrained=True).features

        # freeze all VGG parameters
        for param in self.vgg.parameters():
            param.requires_grad_(False)

        # move the model to GPU, if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vgg.to(self.device)

        #Layer weights for optimization
        self.layerWeights = {'conv1_1': 0.8,
                         'conv2_1': 0.6,
                         'conv3_1': 0.2,
                         'conv4_1': 0.2,
                         'conv5_1': 0.2}

        self.steps = 2000  # for how many iterations to update the image
        
        # for displaying the target image, intermittently
        self.showEvery = 400



    def getFeats(self, img, model, layers=None):
        #Return the outputs at specific model layers
        if layers is None:
            layers = {'0': 'conv1_1',
                      '5': 'conv2_1',
                      '10': 'conv3_1',
                      '19': 'conv4_1',
                      '21': 'conv4_2',
                      '28': 'conv5_1'}

        feats = {}
        x = img
        # model._modules is a dictionary holding each module in the model
        for name, layer in model._modules.items():
            x = layer(x)
            if name in layers:
                feats[layers[name]] = x

        return feats


    def GramMatrix(self, tensor):
        #Calculate the Gram Matrix of a given tensor
        ##batchSize, depth, height, width
        batchSize, d, h, w = tensor.size()
        ## reshape it, so we're multiplying the feats for each channel
        tensor = tensor.view(batchSize * d, h * w)
        ## gram matrix
        gram = torch.mm(tensor, tensor.t())

        return gram

    # function for un-normalizing an image
    # and converting it from a Tensor image to a NumPy image for display
    def ImConvert(self, tensor):
        image = tensor.to("cpu").clone().detach()
        image = image.numpy().squeeze()
        image = image.transpose(1, 2, 0)
        image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
        image = image.clip(0, 1)

        return image

    def transfer(self, content, style):
        # get content and style feats only once before forming the target image
        self.contentFeats = self.getFeats(content, self.vgg)
        self.styleFeats = self.getFeats(style, self.vgg)

        # calculate the gram matrices for each layer of our style representation
        self.styleGrams = {layer: self.GramMatrix(self.styleFeats[layer]) for layer in self.styleFeats}

        # create a third "target" image and prep it for change
        self.target = content.clone().requires_grad_(True).to(self.device)

        self.contentWeight = 1  # alpha
        self.styleWeight = 1e6  # beta

        # iteration hyperparameters
        optimizer = optim.Adam([self.target], lr=0.003)

        for ii in range(1, self.steps + 1):

            ## calculate the content loss
            targetFeats = self.getFeats(self.target, self.vgg)
            contentLoss = torch.mean((targetFeats['conv4_2'] - self.contentFeats['conv4_2']) ** 2)

            # initialize the style loss to 0
            styleLoss = 0
            # iterate through each style layer and add to the style loss
            for layer in self.layerWeights:
                # get the "target" style representation for the layer
                target_feature = targetFeats[layer]

                ## Calculate the target gram matrix
                target_gram = self.GramMatrix(target_feature)
                batchSize, d, h, w = target_feature.shape
                ## Get the "style" style representation
                style_gram = self.styleGrams[layer]
                ## Calculate the style loss for one layer
                layerStyleLoss = self.layerWeights[layer] * torch.mean((target_gram - style_gram) ** 2)

                # add to the style loss
                styleLoss += layerStyleLoss / (d * h * w)

            ## Calculate the total loss
            totalLoss = self.contentWeight * contentLoss + self.styleWeight * styleLoss

            # update target image
            optimizer.zero_grad()
            totalLoss.backward()
            optimizer.step()

            # display intermediate images and print the loss
            if ii % self.showEvery == 0:
                print('Total loss: ', totalLoss.item())
                print('Iteration:   %i/%i' % (ii, self.steps))
                plt.imshow(self.ImConvert(self.target))
                plt.show()

        return self.target

    def changePooling(self, averagePool):
        if averagePool:
            for i, layer in self.vgg.named_children():
                if isinstance(layer, torch.nn.MaxPool2d):
                    self.vgg[int(i)] = torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        else:
            for i, layer in self.vgg.named_children():
                if isinstance(layer, torch.nn.AvgPool2d):
                    self.vgg[int(i)] = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)