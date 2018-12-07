package keanu.example.KeanuExamples.models;

import com.example.keanuexamples.models.ApproximatePi;
import com.example.keanuexamples.utils.SaveAndLoad;
import io.improbable.keanu.network.BayesianNetwork;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.lessThan;
import org.junit.Test;

import java.io.IOException;

public class ApproximatePiTest {
    @Test
    public void testSamplingFromLoadedModelIsMoreAccurateThanSavedModel() throws IOException {
        BayesianNetwork toSaveModel = ApproximatePi.model();
        double lessApproximatePi = ApproximatePi.run(toSaveModel, true);

        BayesianNetwork loadedModel = SaveAndLoad.load(ApproximatePi.FILE_NAME);
        double moreApproximatePi = ApproximatePi.run(loadedModel, false);

        assertThat(Math.abs(Math.PI - moreApproximatePi), lessThan(Math.abs(Math.PI - lessApproximatePi)));
    }
}
