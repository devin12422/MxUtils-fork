#pragma once

#include <QApplication>
#include <QPushButton>
#include <QMainWindow>
#include <QVBoxLayout>
#include <QCommonStyle>
#include <QWidget>
#include <QLabel>
#include <QPixmap>
#include <QImage>
#include <QTimer>
#include <QDebug>
#include <QSignalMapper>
#include <QScreen>
#include <QMenu>
#include <QAction>

#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>

using namespace std;

struct ViewerGeometry
{
    int x;
    int y;
    int w;
    int h;
};

/**
 * @brief DisplayScreen handles multiple FrameViewers.
 * That is, a screen may display multiple streaming channels.
 */
class DisplayScreen : public QWidget
{
    Q_OBJECT

private:
    bool running_ = false;
    friend class FrameViewer;
    uint32_t w_, h_;
    uint32_t num_viewers_;
    vector<QWidget *> viewers_;
    vector<ViewerGeometry> viewer_geometry_;
    void showInferencePopUpMenu(const QPoint &pos);

public:
    DisplayScreen();
    DisplayScreen(QWidget *parent, QScreen *qscreen);
    ~DisplayScreen();

    /**
     * @brief Sets the frame context to be displayed for a viewer given the viewer index.
     *
     * This function sets the frame context to be displayed on a specific viewer
     * identified by the `viewer_id`. It also displays the frames per second (FPS)
     * on the screen.
     *
     * @param viewer_id The index of the viewer.
     * @param frame The frame context represented by a cv::Mat* object.
     * @param fps The FPS number to be shown on the screen.
     */

    void SetDisplayFrame(int viewer_id, cv::Mat *frame, float fps);
    /**
     * @brief Sets the frame context to be displayed for a viewer given the viewer index, without displaying the FPS number.
     *
     * This function sets the frame context to be displayed on a specific viewer
     * identified by the `viewer_id`. The FPS is not displayed on the screen.
     *
     * @param viewer_id The index of the viewer.
     * @param frame The frame context represented by a cv::Mat* object.
     */
    void SetDisplayFrame(int viewer_id, cv::Mat *frame);

    /**
     * @brief Sets the frame context to be displayed for a viewer given the viewer index, without displaying the FPS number.
     *
     * This function sets the frame context to be displayed on a specific viewer
     * identified by the `viewer_id`. The FPS is not displayed on the screen.
     *
     * @param viewer_id The index of the viewer.
     * @param frame The frame context represented by a cv::Mat object.
     */
    void SetDisplayFrame(int viewer_id, cv::Mat frame);

    /**
     * @brief Retrieves the frame context buffer to be displayed for a viewer.
     * @param viewer_id The index of the viewer.
     * @return A pointer to the cv::Mat object representing the frame context buffer.
     */
    cv::Mat *GetDisplayFrameBuf(int viewer_id);

    /**
     * @brief Retrieves the width of the display.
     *
     * This function returns the width of the display screen in pixels.
     *
     * @return The width of the display screen in pixels.
     */
    uint32_t width();

    /**
     * @brief Retrieves the height of the display.
     *
     * This function returns the height of the display screen in pixels.
     *
     * @return The height of the display screen in pixels.
     */
    uint32_t height();

    /**
     * @brief Retrieves the width of a specific viewer's display.
     *
     * This function returns the width of the display for the viewer identified by `viewer_id`.
     *
     * @param viewer_id The index of the viewer.
     * @return The width of the viewer's display in pixels.
     */
    uint32_t GetViewerWidth(int viewer_id);

    /**
     * @brief Retrieves the height of a specific viewer's display.
     *
     * This function returns the height of the display for the viewer identified by `viewer_id`.
     *
     * @param viewer_id The index of the viewer.
     * @return The height of the viewer's display in pixels.
     */
    uint32_t GetViewerHeight(int viewer_id);

    /**
     * @brief Sets a square layout for the viewers.
     *
     * This function arranges the viewers in a square layout based on the number of channels specified.
     *
     * @param num_channels The number of channels to be displayed in the square layout.
     */
    void SetSquareLayout(int num_channels, bool fullscreen = true);

    /**
     * @brief Retrieves the number of viewers.
     *
     * This function returns the total number of viewers currently managed by this screen.
     *
     * @return The number of viewers.
     */
    uint32_t NumViewers();

    /**
     * @brief Adds a new viewer to the display.
     *
     * This function adds a new viewer to the application, allowing it to display content.
     *
     * Advanced usage for user who likes to manually specify the viewer geometry.
     *
     * @param viewer A pointer to the QWidget representing the viewer to be added.
     */
    void AddViewer(QWidget *viewer);

signals:

public slots:
};

/**
 * @brief FrameViewer is responsible for displaying images coming from capture devices.
 * For each channel inside a screen, it should be related to a FrameViewer.
 */
class FrameViewer : public QWidget
{
    Q_OBJECT

public:
    bool running_;
    int idx_;
    int display_frame_idx_;
    vector<cv::Mat *> display_frame_list_;

    FrameViewer(DisplayScreen *parent);
    ~FrameViewer();
    uint32_t width();
    uint32_t height();
    void SetGeometry(int x, int y, int w, int h);
    void SetIdx(int idx);
    void UpdateFrame(cv::Mat *frame);
    void UpdateFPS(float fps);
    void HideFPS();
    void HideChannelName();

signals:
    void signal_UpdateFrame(cv::Mat *);
    void signal_UpdateFPS(float);

public slots:
    void slot_UpdateFrame(cv::Mat *frame);
    void slot_UpdateFPS(float);

private:
    int x_, y_, w_, h_;
    QLabel *frame_;
    QLabel *name_;
    QLabel *fps_;
    QLabel *model_label_;
    QLabel *resolution_label_;
    QLabel *confidence_label_;
};

class MxQt
{
public:
    /**
     * @brief Initialization of screens' information
     * @param argc argument count passed to main() in C/C++
     * @param argv argument vector passed to main() in C/C++
     */
    MxQt(int &argc, char *argv[]);

    // Desctructor
    ~MxQt();

    /**
     * @brief Display the gui and block the current thread until exit.
     */
    int Run();

    /**
     * @brief Number of total available screens
     */
    int num_screens;

    /**
     * @brief Control handle of each screen object
     */
    vector<DisplayScreen *> screens;

private:
    QApplication app;
};
