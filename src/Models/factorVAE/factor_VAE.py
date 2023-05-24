import torch
import torch.nn as nn
from torch.distributions import Normal, kl

from .feature_extractor import FeatureExtractor
from .factor_encoder import FactorEncoder
from .factor_decoder import FactorDecoder
from .factor_predictor import FactorPredictor

class FactorVAE(nn.Module):
    def __init__(
        self,
        characteristic_size,
        stock_size,
        latent_size,
        factor_size,
        time_span,
        gru_input_size,
        hidden_size=64,
        alpha_h_size=64,
    ):
        super(FactorVAE, self).__init__()

        self.characteristic_size = characteristic_size
        self.stock_size = stock_size
        self.latent_size = latent_size

        self.feature_extractor = FeatureExtractor(
            time_span=time_span,
            characteristic_size=characteristic_size,
            latent_size=latent_size,
            stock_size=stock_size,
            gru_input_size=gru_input_size,
        )

        self.factor_encoder = FactorEncoder(
            latent_size=latent_size,
            stock_size=stock_size,
            factor_size=factor_size,
            hidden_size=hidden_size,
        )

        self.factor_decoder = FactorDecoder(
            latent_size=latent_size,
            factor_size=factor_size,
            stock_size=stock_size,
            alpha_h_size=alpha_h_size,
            hidden_size=hidden_size,
        )

        self.factor_predictor = FactorPredictor(
            latent_size=latent_size, factor_size=factor_size, stock_size=stock_size
        )

    def run_model(self, characteristics, future_returns, gamma=1):
        latent_features = self.feature_extractor(characteristics)
        # (batch_size, stock_size, latent_size)
        # print("latent_features:", latent_features.shape)

        mu_post, sigma_post = self.factor_encoder(latent_features, future_returns)
        # (batch_size, factor_size)
        # print("mu_post:", mu_post.shape, "sigma_post:", sigma_post.shape)

        m_encoder = Normal(mu_post, sigma_post)
        factors_post = m_encoder.sample()
        # print("factor_post:", factors_post.shape)

        # (batch_size, factor_size, 1)

        reconstruct_returns, mu_alpha, sigma_alpha, beta = self.factor_decoder(
            factors_post, latent_features
        )
        # print("mu_alpha:", mu_alpha.shape, "sigma_alpha:", sigma_alpha.shape, "beta:", beta.shape)

        mu_dec, sigma_dec = self.get_decoder_distribution(
            mu_alpha, sigma_alpha, mu_post, sigma_post, beta
        )
        # print("mu_dec:", mu_dec.shape, "sigma_dec:", sigma_dec.shape)

        loss_negloglike = (
            Normal(mu_dec, sigma_dec).log_prob(future_returns.unsqueeze(-1)).sum()
        )

        loss_negloglike = loss_negloglike * (-1 / (self.stock_size * latent_features.shape[0]))
        # latent_features.shape[0] is the batch_size

        mu_prior, sigma_prior = self.factor_predictor(latent_features)
        m_predictor = Normal(mu_prior, sigma_prior)
        # print("mu_prior:", mu_prior.shape, "sigma_prior", sigma_prior.shape)

        loss_KL = kl.kl_divergence(m_encoder, m_predictor).sum()

        loss = loss_negloglike + gamma * loss_KL
        # print(f"loss negloglike: {loss_negloglike}, loss_KL: {loss_KL}")

        return loss

    def prediction(self, characteristics):
        with torch.no_grad():

            latent_features = self.feature_extractor(characteristics)

            mu_prior, sigma_prior = self.factor_predictor(latent_features)

            m_prior = Normal(mu_prior, sigma_prior)
            factor_prior = m_prior.sample()

            pred_returns, mu_alpha, sigma_alpha, beta = self.factor_decoder(
                factor_prior, latent_features
            )

            mu_dec, sigma_dec = self.get_decoder_distribution(
                mu_alpha, sigma_alpha, mu_prior, sigma_prior, beta
            )

        return pred_returns, mu_dec, sigma_dec

    def get_decoder_distribution(
        self, mu_alpha, sigma_alpha, mu_factor, sigma_factor, beta
    ):
        # print(mu_alpha.shape, mu_factor.shape, sigma_factor.shape, beta.shape)
        mu_dec = mu_alpha + torch.bmm(beta, mu_factor)

        sigma_dec = torch.sqrt(
            torch.square(sigma_alpha)
            + torch.bmm(torch.square(beta), torch.square(sigma_factor))
        )

        return mu_dec, sigma_dec
