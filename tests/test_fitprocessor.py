#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test functions for :class:`~HAP.utils.fit.FitProcessor`."""

# Third party modules
import numpy as np
import pytest
import random

# Local modules
from CHAP.utils.models import *
from CHAP.utils.fit import FitProcessor


pytestmark = pytest.mark.parametrize('code', ['scipy', 'lmfit'])

@pytest.fixture(scope="class")
def setup_constants(request):
    request.cls.CENTERS = (5.0, 10.0, 15.0)
    request.cls.NUM = 101
    request.cls.X = np.array(np.linspace(-1, 1, request.cls.NUM))
    request.cls.XX = np.array(np.linspace(0, 20, request.cls.NUM))
    request.cls.SIGMA_NOISE = 0.1
    request.cls.SEED = np.random.seed(0)
    request.cls.SIGMA = np.random.normal(
        size=request.cls.NUM, scale=request.cls.SIGMA_NOISE)

def _create_pipelinedata(x, y):
    # Local modules
    from CHAP.pipeline import PipelineData

    return [PipelineData(name='signal', data=y),
            PipelineData(name='coordinates', data=x)]

def _ran_uni():
    return random.random()-0.5

@pytest.mark.usefixtures("setup_constants")
class TestBaseModels:

    def test_exponential(self, code):
        y = exponential(self.X, amplitude=2.0, decay=0.5)
        y += self.SIGMA
        result = FitProcessor.run(
            data=_create_pipelinedata(self.X, y),
            config={
                'code': code,
                'models': [ExponentialModel(model_type='exponential')],
            },
            log_level='WARNING')
        assert pytest.approx(result.redchi) == 4.4905486631e-05

    def test_gaussian1(self, code):
        y = gaussian(self.X, amplitude=2.0, center=0.25, sigma=0.15)
        y += self.SIGMA
        result = FitProcessor.run(
            data=_create_pipelinedata(self.X, y),
            config={
                'code': code,
                'models': [GaussianModel(model_type='gaussian')],
            },
            log_level='WARNING')
        assert pytest.approx(result.redchi) == 3.4482179725e-04

    def test_gaussian2(self, code):
        y = gaussian(self.X, amplitude=2.0, center=0.25, sigma=0.15)
        y += self.SIGMA
        y += linear(self.X, slope=0.2, intercept=-1.5)
        result = FitProcessor.run(
            data=_create_pipelinedata(self.X, y),
            config={
                'code': code,
                'models': [
                    LinearModel(
                        model_type='linear',
                        parameters=[{'name': 'slope', 'value': 0.1}]),
                    GaussianModel(model_type='gaussian')],
            },
            log_level='WARNING')
        assert pytest.approx(result.redchi) == 3.1134959508e-04

    def test_gaussian3(self, code):
        y = gaussian(self.X, amplitude=-2.0, center=0.25, sigma=0.15)
        y += self.SIGMA
        y += parabolic(self.X, a=0.6, b=0.2, c=-1.5)
        result = FitProcessor.run(
            data=_create_pipelinedata(self.X, y),
            config={
                'code': code,
                'models': [
                    QuadraticModel(
                        model_type='parabolic',
                        parameters=[{'name': 'a', 'value': 0.8}]),
                    GaussianModel(
                        model_type='gaussian',
                        parameters=[
                            {'name': 'amplitude', 'value': -1.0},
                            {'name': 'sigma', 'value': 0.25}]),
                ]
            },
            log_level='WARNING')
        assert pytest.approx(result.redchi) == 2.2295871995e-04

    def test_lorentzian(self, code):
        y = lorentzian(self.X, amplitude=2.0, center=0.25, sigma=0.15)
        y += self.SIGMA
        result = FitProcessor.run(
            data=_create_pipelinedata(self.X, y),
            config={
                'code': code,
                'models': [LorentzianModel(model_type='lorentzian')],
            },
            log_level='WARNING')
        assert pytest.approx(result.redchi) == 5.6786804581e-04

    def test_pvoigt(self, code):
        y = pvoigt(
            self.X, amplitude=2.0, center=0.25, sigma=0.15, fraction=0.4)
        y += self.SIGMA
        result = FitProcessor.run(
            data=_create_pipelinedata(self.X, y),
            config={
                'code': code,
                'models': [PseudoVoigtModel(model_type='pvoigt')],
            },
            log_level='WARNING')
        assert pytest.approx(result.redchi) == 3.4043177999e-04

    def test_rectangle1(self, code):
        y = rectangle(
            self.X, amplitude=2.0, center1=-0.5, sigma1=0.1, center2=0.5,
            sigma2=0.05)
        y += self.SIGMA
        result = FitProcessor.run(
            data=_create_pipelinedata(self.X, y),
            config={
                'code': code,
                'models': [
                    RectangleModel(
                        model_type='rectangle',
                        parameters=[
                            {'name': 'center1', 'value': -0.8},
                            {'name': 'sigma1', 'value': 0.2},
                            {'name': 'center2', 'value': 0.7},
                            {'name': 'sigma2', 'value': 0.1}]),
                ],
            },
            log_level='WARNING')
        if code == 'scipy':
            assert pytest.approx(result.redchi) == 1.6945441156e-03
        else:
            assert pytest.approx(result.redchi) == 1.6940805389e-03

    @pytest.mark.parametrize(
        'form, expected', [('atan', 2.59696458705e-03),
                           ('erf', 1.85151985385e-03),
                           ('logistic', 2.0707278023e-03)])
    def test_rectangle2(self, code, form, expected):
        y = rectangle(
            self.X, amplitude=2.0, center1=-0.5, sigma1=0.1, center2=0.5,
            sigma2=0.05, form=form)
        y += self.SIGMA
        result = FitProcessor.run(
            data=_create_pipelinedata(self.X, y),
            config={
                'code': code,
                'models': [
                    RectangleModel(model_type='rectangle', form=form)]},
            log_level='WARNING')
        assert pytest.approx(result.redchi) == expected

    @pytest.mark.parametrize(
        'peak_models, expected', [('gaussian', 1.1039324281e-03),
                                  ('lorentzian', 1.8723852596e-03),
                                  ('pvoigt', 1.1241789925e-03)])
    def test_multipeak(self, code, peak_models, expected):
        random.seed(0)
        model = PEAK_LIKE_MODELS[peak_models](model_type=peak_models)
        kwargs = {'fraction': 0.4} if peak_models == 'pvoigt' else {}
        y = parabolic(self.XX, a=0.002, b=-0.01, c=-0.5)
        y += self.SIGMA
        for center in self.CENTERS:
            y += model.eval(
                self.XX, amplitude=2+3*_ran_uni(), center=center+2*_ran_uni(),
                sigma=0.5+0.2*_ran_uni(), **kwargs)
        result = FitProcessor.run(
            data=_create_pipelinedata(self.XX, y),
            config={
                'code': code,
                'models': [
                    QuadraticModel(model_type='parabolic'),
                    MultipeakModel(
                        model_type='multipeak', centers=self.CENTERS,
                        peak_models=peak_models),
                ],
            },
            log_level='WARNING')
        assert pytest.approx(result.redchi) == expected

    def test_expression1(self, code):
        random.seed(0)
        expr = ''
        y = parabolic(self.XX, a=0.002, b=-0.01, c=-0.5)
        y += self.SIGMA
        for i, center in enumerate(self.CENTERS):
            y += gaussian(
                self.XX, amplitude=2+3*_ran_uni(), center=center+2*_ran_uni(),
                sigma=0.5+0.2*_ran_uni())
            if i:
                expr += ' + '
            expr += f'amp{i+1}/(2.5066282746310002*sig{i+1}) * ' \
                    f'exp(-(x-cen{i+1})**2 / max(1e-15, (2*sig{i+1}**2)))' 
        result = FitProcessor.run(
            data=_create_pipelinedata(self.XX, y),
            config={
                'code': code,
                'models': [
                    QuadraticModel(model_type='parabolic'),
                    ExpressionModel(
                        model_type='expression',
                        expr=expr,
                        parameters=[
                            {'name': 'cen1', 'value': self.CENTERS[0]-1},
                            {'name': 'sig1', 'min': 0.05, 'max': 3.0},
                            {'name': 'cen2', 'value': self.CENTERS[1]+0.5},
                            {'name': 'sig2', 'min': 0.05, 'max': 3.0},
                            {'name': 'cen3', 'value': self.CENTERS[2]+1.5},
                            {'name': 'sig3', 'min': 0.05, 'max': 3.0}]),
                ],
            },
            log_level='WARNING')
        assert pytest.approx(result.redchi) == 1.1039324286e-03

    def test_expression2(self, code):
        random.seed(0)
        b = 0.96
        c = -0.05
        y = linear(self.XX, slope=0.02, intercept=0.5)
        y += self.SIGMA
        models = [LinearModel(model_type='linear')]
        for i, center in enumerate(self.CENTERS):
            y += gaussian(
                self.XX, amplitude=2+3*_ran_uni(),
                center=(center-(c+0.02*_ran_uni()))/(b+0.02*_ran_uni()),
                sigma=0.5+0.2*_ran_uni())
            models.append(
                GaussianModel(
                    model_type='gaussian',
                    parameters=[
                        {'name': 'amplitude', 'min': 1.e-15},
                        {'name': 'center', 'expr': f'({self.CENTERS[i]}-c)/b'},
                        {'name': 'sigma', 'min': 0.05, 'max': 3.0}]),
            )
        result = FitProcessor.run(
            data=_create_pipelinedata(self.XX, y),
            config={
                'code': code,
                'models': models,
                'parameters': [{'name': 'b', 'value': 0.99, 'min': 0.9,
                                'max': 1.1},
                               {'name': 'c', 'value': 0.01}],
            },
            log_level='WARNING')
#        print(f'redchi: {result.redchi:.10e}')
#        result.print_fit_report()
#        result.plot(plot_comp_legends=True)
        assert pytest.approx(result.redchi) == 1.3062197542e-03
