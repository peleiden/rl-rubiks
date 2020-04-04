import { Component } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
})
export class AppComponent {
  constructor(private http: HttpClient) {
    http.post("http://127.0.0.1:5000/cube", []).toPromise().then(val => console.log(val));
  }
}
