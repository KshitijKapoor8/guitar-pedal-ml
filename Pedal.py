import torch
import torch.nn as nn
import math
from torch.utils.data import DataLoader, Dataset
from contextlib import nullcontext


class Pedal(nn.Module):
    # determine how many channels our audio data represents. 1 for mono 2 for stereo
    # should be the same for input and output
    INPUT_OUTPUT_SIZE = 1

    def __init__(
        self,
        hidden_size,
        window_size,
        normalization_data,
        init_len=200,
        num_layers=1,
        hidden=None,
    ):
        super(Pedal, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.window_size = window_size
        self.init_len = init_len

        self.lstm = nn.LSTM(Pedal.INPUT_OUTPUT_SIZE, hidden_size, self.num_layers)
        self.last_layer = nn.Linear(hidden_size, Pedal.INPUT_OUTPUT_SIZE, bias=True)
        self.hidden = hidden

        self.normalization_data = normalization_data

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, self.hidden = self.lstm(x, self.hidden)
        output = self.last_layer(output)

        return output + x

    def train_epoch(
        self,
        data_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        self.train()
        batch_loss = 0.0
        for inputs, targets in data_loader:
            # populate hidden layer with information using some early data
            init_inputs = inputs[0 : self.init_len :, :, :]
            rest_inputs = inputs[self.init_len :, :, :]
            rest_targets = targets[self.init_len :, :, :]

            self(init_inputs)

            window_start_idx = self.init_len
            window_loss = 0

            # so that we don't have zero range
            window_iterations = (
                len(rest_inputs) // self.window_size
                if len(rest_inputs) % self.window_size == 0
                else 1
            )
            for _ in range(window_iterations):
                window_inputs = rest_inputs[
                    window_start_idx : window_start_idx + self.window_size, :, :
                ]
                window_targets = rest_targets[
                    window_start_idx : window_start_idx + self.window_size, :, :
                ]

                outputs = self(window_inputs)
                loss = criterion(outputs, window_targets)

                loss.backward()
                optimizer.step()

                # prepare for next batch
                # save hidden state for next batch
                self.detach_hidden()
                self.zero_grad()
                window_loss += loss
                window_start_idx += self.window_size

            batch_loss += (
                window_loss / window_iterations if window_iterations > 0 else 0
            )
            self.reset_hidden()

        return batch_loss / len(data_loader)

    def train_model(
        self,
        data_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
        num_epochs: int,
        save_path: str,
    ) -> float:
        self.train()
        lowest_loss = float("inf")
        for epoch in range(num_epochs):
            loss = self.train_epoch(data_loader, criterion, optimizer)
            scheduler.step(loss)
            print(f"Epoch: {epoch + 1}, Loss: {loss}")

            lowest_loss = min(lowest_loss, loss)
            if loss == lowest_loss:
                torch.save(self.state_dict(), save_path)
                print("Model saved successfully.")

        print(f"Lowest Loss: {lowest_loss} reached after {num_epochs} epochs.")
        return lowest_loss

    def normalize_data(self, data: torch.Tensor) -> torch.Tensor:
        return (data - self.normalization_data["min"]) / (
            self.normalization_data["max"] - self.normalization_data["min"]
        )

    # detach hidden state, this resets gradient tracking on the hidden state
    def detach_hidden(self):
        if self.hidden.__class__ == tuple:
            self.hidden = tuple([h.clone().detach() for h in self.hidden])
        else:
            self.hidden = self.hidden.clone().detach()

    # changes the hidden state to None, causing pytorch to create an all-zero hidden state when the rec unit is called
    def reset_hidden(self):
        self.hidden = None

    def save_model(self, path: str) -> None:
        torch.save(self.state_dict(), path)
        # print("Model saved successfully.")

    def load_model(self, path: str) -> None:
        self.load_state_dict(torch.load(path))

    def predict(self, x: torch.Tensor, normalize=False, num_batches=1) -> torch.Tensor:
        self.eval()

        if normalize:
            x = self.normalize_data(x)

        # Reshape input tensor to LSTM input shape
        x = x.unsqueeze(1)

        # Batch prediction
        batch_size = x.shape[0] // num_batches

        output = torch.zeros_like(x)

        with torch.no_grad():
            for i in range(num_batches):
                output[i * batch_size : (i + 1) * batch_size] = self(
                    x[i * batch_size : (i + 1) * batch_size]
                )
                self.detach_hidden()

            if num_batches * batch_size < x.shape[0]:
                output[num_batches * batch_size :] = self(
                    x[num_batches * batch_size :]
                )

            self.reset_hidden()

        return output

    def process_data(self, input_data: torch.Tensor, target_data: torch.Tensor, criterion, num_batches, normalize=False) -> torch.Tensor:
        self.eval()

        if normalize:
            input_data = self.normalize_data(input_data)

        # Reshape input tensor to LSTM input shape
        input_data = input_data.unsqueeze(1)

        # Batch prediction
        batch_size = input_data.shape[0] // num_batches

        output = torch.zeros_like(input_data)

        total_loss = 0.0
        with torch.no_grad():
            for i in range(num_batches):
                output[i * batch_size : (i + 1) * batch_size] = self(
                    input_data[i * batch_size : (i + 1) * batch_size]
                )
                self.detach_hidden()

                total_loss += criterion(output[i * batch_size : (i + 1) * batch_size], target_data[i * batch_size : (i + 1) * batch_size])

            if num_batches * batch_size < input_data.shape[0]:
                output[num_batches * batch_size :] = self(
                    input_data[num_batches * batch_size :]
                )

                total_loss += criterion(output[num_batches * batch_size :], target_data[num_batches * batch_size :])

            self.reset_hidden()

        return output, total_loss / num_batches