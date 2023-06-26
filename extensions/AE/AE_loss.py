import torch
import os
import time
from torch.autograd import Function
from torch import nn
from extensions.AE._ext import my_lib

class AElossFunction(Function):
    def forward(self, tags, keypoints):
        output = torch.zeros(torch.Size((tags.size()[0], 2)))
        mean_tags = torch.zeros(torch.Size((tags.size()[0], keypoints.size()[1], tags.size()[2]+1)))
        self.mean_tags = mean_tags

        my_lib.my_lib_loss_forward(tags, keypoints, output, mean_tags)
        self.save_for_backward(tags, keypoints)
        return output

    def backward(self, grad_output):
        tags, keypoints = self.saved_tensors
        grad_input = torch.zeros(tags.size()).cuda(tags.get_device())
        #grad_input = tags.new(tags.size()).zero_()
        my_lib.my_lib_loss_backward(tags, keypoints, self.mean_tags, grad_output, grad_input)
        self.mean_tags = None
        return grad_input, torch.zeros(keypoints.size())

class AEloss(nn.Module):
    def forward(self, input, input1):
        if not input.is_cuda:
            input = input.cuda()
        output = AElossFunction()(input, input1)
        return output


class AEloss1(nn.Module):
    def forward(self, tags, keypoints):
        """
        Inputs:
            tags: [batch_size, 17 * output_res * output_res, tag_dim]
            keypoints: [batch_size, max_num_people, 17, 2]
        Return:
            output: [batch_size, 2]
                output[:, 0]: push loss
                output[:, 1]: pull loss
        """
        pushes, pulls = [], []
        batch_size = tags.size(0)
        for i in range(batch_size):
            push, pull = self.single_image_tag_loss(tags[i], keypoints[i])
            pushes.append(push.reshape(1))
            pulls.append(pull.reshape(1))
        return torch.cat([torch.stack(pushes), torch.stack(pulls)], dim=1)

    def single_image_tag_loss(self, tags, keypoints):
        """
        Inputs:
            tag: [17 * output_res * output_res, tag_dim]
            keypoints: [max_num_people, 17, 2]
        """
        eps = 1e-6
        pull = 0
        tag_dim = tags.size(1)
        mean_tags = []
        for keypoints_person in keypoints:
            mask = keypoints_person[:, 1] > 0
            tags_person = tags[keypoints_person[mask][:, 0].long()]
            if tags_person.size(0) == 0:
                continue
            mean_tags.append(torch.mean(tags_person, dim=0))
            pull += torch.mean(torch.pow(tags_person - mean_tags[-1], 2).sum(1))

        if len(mean_tags) == 0:
            return torch.zeros([1]).cuda(), torch.zeros([1]).cuda()

        mean_tags = torch.stack(mean_tags)  # [person_num, tag_dim]
        person_num = mean_tags.size(0)

        x = mean_tags.unsqueeze(1).expand(person_num, person_num, tag_dim)
        diff = torch.pow(x - x.permute(1, 0, 2), 2).sum(2)  # [person_num, person_num]
        upper_triangle_idx = diff.triu(1).nonzero()
        diff = diff[upper_triangle_idx[:, 0], upper_triangle_idx[:, 1]]
        push = torch.exp(-diff).sum()
        return push / ((person_num - 1) * person_num + eps), pull / (person_num + eps)
