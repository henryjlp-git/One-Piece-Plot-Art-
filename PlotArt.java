import javax.swing.*;
import java.awt.*;
import java.util.ArrayList;
import java.util.List;

/**
 * Program
 * Description
 * Section
 * Date
 * @author Student Name
 */

// Abstract class for all drawable elements
abstract class DrawableElement {
    protected int x, y;

    public DrawableElement(int x, int y) {
        this.x = x;
        this.y = y;
    }

    public abstract void draw(Graphics g);
}

/*
You will need to create at least 3 different classes that inherit from
DrawableElement and override the draw method. Your new elements should
create a simple image of your choosing. Your image must include at
least one instance of each element, with at least 5 instances total.

An image that uses at least 6 different elements with at least 10
instances will receive extra credit. Have fun creating your art!
 */

// PlotArt class
public class PlotArt extends JFrame {
    private final List<DrawableElement> elements;

    public PlotArt() {
        elements = new ArrayList<>();
        //Add elements to be plotted

        setSize(800, 600);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setVisible(true);
    }

    @Override
    public void paint(Graphics g) {
        super.paint(g);
        for (DrawableElement element : elements) {
            element.draw(g);
        }
    }

    public static void main(String[] args) {
        new PlotArt();
    }
}
