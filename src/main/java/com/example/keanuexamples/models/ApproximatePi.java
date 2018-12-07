package com.example.keanuexamples.models;

import com.example.keanuexamples.utils.SaveAndLoad;
import io.improbable.keanu.algorithms.NetworkSamples;
import io.improbable.keanu.algorithms.mcmc.MetropolisHastings;
import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.tensor.dbl.DoubleTensor;
import io.improbable.keanu.tensor.dbl.ScalarDoubleTensor;
import io.improbable.keanu.vertices.Vertex;
import io.improbable.keanu.vertices.VertexLabel;
import io.improbable.keanu.vertices.dbl.DoubleVertex;
import io.improbable.keanu.vertices.dbl.DoubleVertexSamples;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.ConstantDoubleVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.AdditionVertex;
import io.improbable.keanu.vertices.dbl.nonprobabilistic.operators.binary.PowerVertex;
import io.improbable.keanu.vertices.dbl.probabilistic.SmoothUniformVertex;
import lombok.experimental.UtilityClass;

import java.io.IOException;
import java.util.Collections;

/**
 * @see <a href="http://www.eveandersson.com/pi/monte-carlo-circle">Calculation of Pi Using the Monte Carlo Method</a>
 */
@UtilityClass
public class ApproximatePi {

    private static final int RADIUS = 100000;
    private static final int SAMPLE_COUNT = 100000;
    private static final String X_SQUARED_PLUS_Y_SQUARED_LABEL = "x_squared_plus_y_squared";
    public static final String FILE_NAME = "approximatepi.proto";

    public double run(BayesianNetwork network, boolean save) throws IOException {
        DoubleVertex xSquaredPlusYSquared = (DoubleVertex) network.getVertexByLabel(new VertexLabel(X_SQUARED_PLUS_Y_SQUARED_LABEL));
        NetworkSamples networkSamples = MetropolisHastings.withDefaultConfig().getPosteriorSamples(network, Collections.singletonList(xSquaredPlusYSquared), SAMPLE_COUNT);

        if (save) {
            SaveAndLoad.save(network, FILE_NAME);
        }

        DoubleVertexSamples result = networkSamples.getDoubleTensorSamples(xSquaredPlusYSquared);
        ScalarDoubleTensor rSquared = new ScalarDoubleTensor(Math.pow(RADIUS, 2));
        double probability = result.probability(doubleTensor -> isInCircle(doubleTensor, rSquared));
        return probability * 4;
    }

    public BayesianNetwork model() {
        SmoothUniformVertex x = new SmoothUniformVertex(0.0, RADIUS);
        SmoothUniformVertex y = new SmoothUniformVertex(0.0, RADIUS);

        PowerVertex xSquared = new PowerVertex(x, new ConstantDoubleVertex(2));
        PowerVertex ySquared = new PowerVertex(y, new ConstantDoubleVertex(2));

        AdditionVertex xSquaredPlusYSquared = new AdditionVertex(xSquared, ySquared);
        xSquaredPlusYSquared.setLabel(X_SQUARED_PLUS_Y_SQUARED_LABEL);

        return new BayesianNetwork(xSquaredPlusYSquared.getConnectedGraph());
    }

    private boolean isInCircle(DoubleTensor sum, ScalarDoubleTensor rSquared) {
        return sum.lessThan(rSquared).allTrue();
    }
}