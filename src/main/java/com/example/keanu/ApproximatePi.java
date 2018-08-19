package com.example.keanu;

import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.mcmc.MetropolisHastings;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.dbl.ScalarDoubleTensor;
import io.improbable.keanu.vertices.dbl.DoubleVertexSamples;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.AdditionVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.PowerVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.SmoothUniformVertex;

import java.util.Collections;

public class ApproximatePi {

    public double run(int min, int max, int sampleCount) {
        SmoothUniformVertex x = new SmoothUniformVertex(min, max);
        SmoothUniformVertex y = new SmoothUniformVertex(min, max);

        PowerVertex xSquared = new PowerVertex(x, new ConstantDoubleVertex(2));
        PowerVertex ySquared = new PowerVertex(y, new ConstantDoubleVertex(2));
        ScalarDoubleTensor rSquared = new ScalarDoubleTensor(Math.pow(max - min, 2));

        AdditionVertex xSquaredPlusYSquared = new AdditionVertex(xSquared, ySquared);

        BayesianNetwork network = new BayesianNetwork(xSquaredPlusYSquared.getConnectedGraph());
        NetworkSamples networkSamples = MetropolisHastings.getPosteriorSamples(network, Collections.singletonList(xSquaredPlusYSquared), sampleCount);

        DoubleVertexSamples result = networkSamples.getDoubleTensorSamples(xSquaredPlusYSquared);
        double probability = result.probability(doubleTensor -> isInCircle(doubleTensor, rSquared));

        return probability * 4;
    }

    private boolean isInCircle(DoubleTensor sum, ScalarDoubleTensor rSquared) {
        return sum.lessThan(rSquared).allTrue();
    }
}