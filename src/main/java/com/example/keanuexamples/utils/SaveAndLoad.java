package com.example.keanuexamples.utils;

import io.improbable.keanu.network.BayesianNetwork;
import io.improbable.keanu.network.ProtobufLoader;
import io.improbable.keanu.network.ProtobufSaver;
import lombok.experimental.UtilityClass;
import org.apache.commons.io.FileUtils;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;


@UtilityClass
public class SaveAndLoad {

    public void save(BayesianNetwork network, String fileName) throws IOException {
        ByteArrayOutputStream output = new ByteArrayOutputStream();
        ProtobufSaver saver = new ProtobufSaver(network);
        saver.save(output, true);
        output.writeTo(new FileOutputStream(createResultFile(fileName)));
    }

    public BayesianNetwork load(String fileName) throws IOException {
        ByteArrayInputStream input = new ByteArrayInputStream(FileUtils.readFileToByteArray(createResultFile(fileName)));
        ProtobufLoader loader = new ProtobufLoader();
        return loader.loadNetwork(input);
    }

    private static File createResultFile(String fileName) {
        String currentDirectory = System.getProperty("user.dir");
        return new File(currentDirectory + "/results/" + fileName);
    }
}
