#include <fstream>
#include <string>
#include <iostream>
using namespace std;

void get_memory_size(string message) {
    ifstream file;
    file.open("/proc/self/status");
    string line;

    int numberOfLinesToRead = 4;
    int linesRead = 0;

    cout << message << endl;

    while (getline(file, line)) {
        linesRead++;
        
        if (linesRead == 18 || linesRead == 22)
        cout << line << endl;
    }
}