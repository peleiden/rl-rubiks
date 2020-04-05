import { Component, OnInit } from '@angular/core';
import { HttpService } from './http.service';
import { CommonService } from './common.service';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styles: [
    "button.btn-action { width: 71px; }"
  ],
})
export class AppComponent implements OnInit {

  scrambleDepth = 10;
  actions = ["F", "f", "B", "b", "T", "t", "D", "d", "L", "l", "R", "r"];

  constructor(private httpService: HttpService, public commonService: CommonService) { }

  ngOnInit() {
    this.commonService.getSolved();
  }
  // TODO: Før oh med over alt
}
